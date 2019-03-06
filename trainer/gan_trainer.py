import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from torchvision import transforms
from base import BaseGANTrainer
from model.damsm import DAMSM
from matplotlib import pyplot as plt


dirname = os.path.dirname(__file__)
main_dirname = os.path.dirname(dirname)
birds_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-Birds/0226_204228/model_best.pth')
flowers_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-Flowers/0226_204709/model_best.pth')
coco_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-CoCo/0226_180051/model_best.pth')


class Trainer(BaseGANTrainer):
    """
    Example Trainer class
    Note:
        Inherited from BaseTrainer.
    """

    def __init__(self, models, optimizers, losses, metrics, resume, config,
                 data_loader, valid_data_loader=None, train_logger=None):

        super(Trainer, self).__init__(
            generator=models["generator"],
            discriminator=models["discriminator"],
            generator_optimizer=optimizers["generator"],
            discriminator_optimizer=optimizers["discriminator"],
            losses=losses,
            metrics=metrics,
            resume=resume,
            config=config,
            train_logger=train_logger)

        self.config = config
        self.train_data_loader = data_loader
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.image_size = self.config["train_data_loader"]["args"]["image_size"]
        self.noise_dim = self.config["models"]["Generator"]["args"]["noise_dim"]

        print("load damsm ecoding model")
        damsm = DAMSM(
            vocab_size=len(self.train_data_loader.dataset.vocab),
            word_embed_size=512,
            embedding_size=1024)
        if "Bird" in self.config["name"]:
            resume_path = birds_damsm
        elif "Flower" in self.config["name"]:
            resume_path = flowers_damsm
        elif "CoCo" in self.config["name"]:
            resume_path = coco_damsm
        else:
            raise ValueError("cannot find corresponding damsm model path")
        checkpoint = torch.load(resume_path, map_location=self.device)
        damsm.load_state_dict(checkpoint["state_dict"])
        for p in damsm.parameters():
            p.requires_grad = False
        self.damsm_rnn_encoder = damsm.rnn_encoder
        self.damsm_rnn_encoder.to(self.device)
        if len(self.device_ids) > 1:
            self.damsm_rnn_encoder = torch.nn.DataParallel(self.damsm_rnn_encoder, device_ids=self.device_ids)


    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
            The metrics in log must have the key 'metrics'.
        """
        self.generator.train()
        self.discriminator.train()

        total_generator_loss = 0
        total_discriminator_loss = 0

        for batch_idx, data in enumerate(self.train_data_loader):
            # right image
            right_images = {}
            right_images['256'] = data["right_images_{}".format(self.image_size)].to(self.device)
            right_images['128'] = data["right_images_128"].to(self.device)
            right_images['64'] = data["right_images_64"].to(self.device)
            right_captions = data["right_captions"].to(self.device)
            right_caption_lengths = data["right_caption_lengths"].to(self.device)
            _, right_embeddings = self.damsm_rnn_encoder(right_captions, right_caption_lengths)
            right_embeddings.to(self.device)

            # other image
            wrong_images = {}
            wrong_images['256'] = data["wrong_images_256"].to(self.device)
            wrong_images['128'] = data["wrong_images_128"].to(self.device)
            wrong_images['64'] = data["wrong_images_64"].to(self.device)

            # train the generator first
            for p in self.discriminator.parameters():
                p.prequires_grad = False
            for p in self.generator.parameters():
                p.prequires_grad = True
            self.generator_optimizer.zero_grad()

            noise = Variable(torch.randn(right_images['256'].size(0), self.noise_dim)).to(self.device)
            noise = noise.view(noise.size(0), self.noise_dim)
            generated_images, mean_var = self.to_img_dict_(self.generator(right_embeddings, noise))

            generator_loss = 0
            #---- iterate over image of different sizes ----#
            '''Compute gen loss'''
            for key, _ in generated_images.items():
                this_fake = generated_images[key]
                fake_pair_logit, fake_img_logit_local = self.discriminator(this_fake, right_embeddings)

                # -- compute pair loss ---
                real_global_labels = torch.ones(fake_pair_logit.size(0))
                real_labels = Variable(real_global_labels).to(self.device)
                pair_loss = self.compute_g_loss(fake_pair_logit, real_labels)
                generator_loss += pair_loss

                # -- compute image loss ---
                real_local_labels = torch.ones((right_images['256'].size(0), 1, 5, 5))
                real_labels = Variable(real_local_labels).to(self.device)
                img_loss = self.compute_g_loss(fake_img_logit_local, real_labels)
                generator_loss += img_loss

            # KL loss
            kl_loss = self.get_KL_Loss(mean_var[0], mean_var[1])
            generator_loss += self.config["trainer"]["KL_coe"] * kl_loss

            generator_loss.backward()
            self.generator_optimizer.step()

            # train the discriminator
            for p in self.discriminator.parameters():
                p.prequires_grad = True
            for p in self.generator.parameters():
                p.prequires_grad = False
            self.discriminator_optimizer.zero_grad()

            noise = Variable(torch.randn(right_images['256'].size(0), self.noise_dim)).to(self.device)
            noise = noise.view(noise.size(0), self.noise_dim)
            fake_images_64, fake_images_128, fake_images_256, mean, logsigma = self.generator(right_embeddings, noise)
            # fake_images, mean_var = self.to_img_dict_(self.generator(right_embeddings, noise))
            fake_images, mean_var = self.to_img_dict_(fake_images_64.detach(), fake_images_128.detach(), fake_images_256.detach(), mean.detach(), logsigma.detach())

            discriminator_loss = 0
            ''' iterate over image of different sizes.'''
            for key, _ in fake_images.items():
                this_img = right_images[key]
                this_wrong = wrong_images[key]
                this_fake = Variable(fake_images[key].data)

                real_logit, real_img_logit_local = self.discriminator(this_img, right_embeddings)
                wrong_logit, wrong_img_logit_local = self.discriminator(this_wrong, right_embeddings)
                fake_logit, fake_img_logit_local = self.discriminator(this_fake, right_embeddings)

                ''' compute disc pair loss '''
                real_global_labels = torch.ones(real_logit.size(0))
                fake_global_labels = -torch.ones(real_logit.size(0))
                real_labels = Variable(real_global_labels).to(self.device)
                fake_labels = Variable(fake_global_labels).to(self.device)
                pair_loss = self.compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)
                discriminator_loss += pair_loss

                ''' compute disc image loss '''
                real_local_labels = torch.ones((right_images['256'].size(0), 1, 5, 5))
                fake_local_labels = -torch.ones((right_images['256'].size(0), 1, 5, 5))
                real_labels = Variable(real_local_labels).to(self.device)
                fake_labels = Variable(fake_local_labels).to(self.device)
                img_loss = self.compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local,
                                              real_labels, fake_labels)
                discriminator_loss += img_loss

            discriminator_loss.backward()
            self.discriminator_optimizer.step()

            self.writer.set_step((epoch - 1) * len(self.train_data_loader) + batch_idx)
            self.writer.add_scalar('Generator_Loss', generator_loss.item())
            self.writer.add_scalar('Discriminator_Loss', discriminator_loss.item())
            total_generator_loss += generator_loss.item()
            total_discriminator_loss += discriminator_loss.item()

            if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] '
                                 'Generator Loss: {:.6f} '
                                 'Discriminator Loss: {:.6f}'.format(
                    epoch,
                    batch_idx * self.train_data_loader.batch_size,
                    self.train_data_loader.n_samples,
                    100.0 * batch_idx / len(self.train_data_loader),
                    generator_loss.item(),
                    discriminator_loss.item()
                ))

        self.writer.add_image('input', make_grid(generated_images['256'].cpu(), nrow=8, normalize=True))
        if epoch%5 == 0:
            self.predict(self.valid_data_loader, epoch)

        log = {
            'Generator_Loss': total_generator_loss / len(self.train_data_loader),
            'Discriminator_Loss': total_discriminator_loss / len(self.train_data_loader),
            'metrics': None
        }

        return log

    def _valid_epoch(self, epoch):
        pass

    def _train_generator_epoch(self, epoch):
        pass

    def _train_discriminator_epoch(self, epoch):
        pass

    def compute_d_pair_loss(self, real_logit, wrong_logit, fake_logit, real_labels, fake_labels):
        criterion = self.losses
        real_d_loss = criterion(real_logit, real_labels)
        wrong_d_loss = criterion(wrong_logit, fake_labels)
        fake_d_loss = criterion(fake_logit, fake_labels)

        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        return discriminator_loss

    def compute_d_img_loss(self, wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):
        criterion = self.losses
        wrong_d_loss = criterion(wrong_img_logit, real_labels)
        real_d_loss = criterion(real_img_logit, real_labels)
        fake_d_loss = criterion(fake_img_logit, fake_labels)

        return fake_d_loss + (wrong_d_loss + real_d_loss) / 2

    def compute_g_loss(self, fake_logit, real_labels):
        criterion = self.losses
        generator_loss = criterion(fake_logit, real_labels)
        return generator_loss

    @staticmethod
    def to_img_dict_(*inputs):
        if type(inputs[0]) == tuple:
            inputs = inputs[0]
        res = {}
        res['64'] = inputs[0]
        res['128'] = inputs[1]
        res['256'] = inputs[2]
        mean_var = (inputs[3], inputs[4])
        loss = mean_var
        return res, loss

    @staticmethod
    def get_KL_Loss(mu, logvar):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
        kl_loss = torch.mean(kld)
        return kl_loss

    def predict(self, data_loader, epoch=None):

        self.generator.eval()
        self.discriminator.eval()

        mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
        std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        transform = transforms.Compose([
            transforms.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist()),
            transforms.ToPILImage()]
        )

        for batch_idx, sample in enumerate(data_loader):
            if batch_idx == 2:
                break

            right_images_256 = sample['right_images_256'].to(self.device)
            right_images_128 = sample['right_images_128'].to(self.device)
            right_images_64 = sample['right_images_64'].to(self.device)
            right_captions = sample["right_captions"].to(self.device)
            right_caption_lengths = sample["right_caption_lengths"].to(self.device)
            _, right_embeddings = self.damsm_rnn_encoder(right_captions, right_caption_lengths)
            right_embeddings.to(self.device)
            txt = sample['right_txt']

            if not os.path.exists('{0}/results/epoch_{1}'.format(self.checkpoint_dir, epoch)):
                os.makedirs('{0}/results/epoch_{1}'.format(self.checkpoint_dir, epoch))

            noise = Variable(torch.randn(right_images_256.size(0), self.noise_dim)).to(self.device)
            noise = noise.view(noise.size(0), self.noise_dim)

            fake_images_64, fake_images_128, fake_images_256, _, _ = self.generator(right_embeddings, noise)
            real_logit_64, _ = self.discriminator(right_images_64, right_embeddings)
            real_logit_128, _ = self.discriminator(right_images_128, right_embeddings)
            real_logit_256, _ = self.discriminator(right_images_256, right_embeddings)
            fake_logit_64, _ = self.discriminator(fake_images_64, right_embeddings)
            fake_logit_128, _ = self.discriminator(fake_images_128, right_embeddings)
            fake_logit_256, _ = self.discriminator(fake_images_256, right_embeddings)

            cnt = 0
            for f_64, f_128, f_256, r_64, r_128, r_256, t in zip(fake_images_64, fake_images_128, fake_images_256, right_images_64, right_images_128, right_images_256, txt):
                fig, axs = plt.subplots(3, 2)

                axs[0, 0].set_title('Fake 64 Disc:{:.1f}'.format(fake_logit_64[cnt][0]))
                axs[0, 0].imshow(np.array(transform(f_64.cpu())))
                axs[0, 0].axis("off")
                axs[1, 0].set_title('Fake 128 Disc:{:.1f}'.format(fake_logit_128[cnt][0]))
                axs[1, 0].imshow(np.array(transform(f_128.cpu())))
                axs[1, 0].axis("off")
                axs[2, 0].set_title('Fake 256 Disc:{:.1f}'.format(fake_logit_256[cnt][0]))
                axs[2, 0].imshow(np.array(transform(f_256.cpu())))
                axs[2, 0].axis("off")

                axs[0, 1].set_title('Real 64 Disc:{:.1f}'.format(real_logit_64[cnt][0]))
                axs[0, 1].imshow(np.array(transform(r_64.cpu())))
                axs[0, 1].axis("off")
                axs[1, 1].set_title('Real 128 Disc:{:.1f}'.format(real_logit_128[cnt][0]))
                axs[1, 1].imshow(np.array(transform(r_128.cpu())))
                axs[1, 1].axis("off")
                axs[2, 1].set_title('Real 256 Disc:{:.1f}'.format(real_logit_256[cnt][0]))
                axs[2, 1].imshow(np.array(transform(r_256.cpu())))
                axs[2, 1].axis("off")
                fig.savefig('{0}/results/epoch_{1}/{2}.jpg'.format(self.checkpoint_dir, epoch, t.replace("/", "")[:100]))

