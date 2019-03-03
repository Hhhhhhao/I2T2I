import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision.utils import make_grid
from torch.nn.utils.rnn import pack_padded_sequence
from base import BaseGANTrainer
from utils import get_caption_lengths
from model.damsm import DAMSM
from utils.plot_utils import *
from utils.util import to_numpy

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
        self.noise_dim = self.config["models"]["generator"]["args"]["noise_dim"]

        print("load damsm ecoding model")
        self.damsm = DAMSM(
            vocab_size=self.train_data_loader.dataset.vocab,
            word_embed_size=512,
            embedding_size=1024)
        if "bird" in self.config.name:
            resume_path = birds_damsm
        elif "flower" in self.config.name:
            resume_path = flowers_damsm
        elif "CoCo" in self.config.name:
            resume_path = coco_damsm
        else:
            raise ValueError("cannot find corresponding damsm model path")
        checkpoint = torch.load(resume_path)
        self.damsm.load_state_dict(checkpoint["state_dict"])

    def init_plots(self):
        # -------------init ploters for losses----------------------------#
        display_freq = 200 # update loss every 200 batches
        self.d_loss_plot = plot_scalar(
            name="d_loss", env=self.config.name, rate=display_freq)
        self.g_loss_plot = plot_scalar(
            name="g_loss", env=self.config.name, rate=display_freq)
        self.kl_loss_plot = plot_scalar(name="kl_loss", env=self.config.name, rate=display_freq)

        all_keys = ["64", "128", "256"]
        self.g_plot_dict, self.d_plot_dict = {}, {}
        for this_key in all_keys:
            self.g_plot_dict[this_key] = plot_scalar(
                name="g_img_loss_" + this_key, env=self.config.name, rate=display_freq)
            self.d_plot_dict[this_key] = plot_scalar(
                name="d_img_loss_" + this_key, env=self.config.name, rate=display_freq)

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
            right_caption_lengths = data["right_caption_lengths"]
            right_embeddings = self.damsm.rnn_encoder(right_captions, right_caption_lengths)
            right_embeddings.to(self.device)

            # other image
            wrong_images = {}
            wrong_images['256'] = data["wrong_images_256"].to(self.device)
            wrong_images['128'] = data["wrong_images_128"].to(self.device)
            wrong_images['64'] = data["wrong_images_64"].to(self.device)

            # train the generator first
            self.generator.unfreeze()
            self.discriminator.freeze()
            self.generator_optimizer.zero_grad()

            noise = Variable(torch.randn(right_images['256'].size(0), self.noise_dim)).to(self.device)
            noise = noise.view(noise.size(0), self.noise_dim, 1, 1)
            generated_images, mean_var = self.to_img_dict_(self.generator(right_embeddings, noise))

            generator_loss = 0
            #---- iterate over image of different sizes ----#
            '''Compute gen loss'''
            for key, _ in generated_images.items():
                this_fake = generated_images[key]
                fake_pair_logit, fake_img_logit_local = self.discriminator(this_fake, right_embeddings)

                # -- compute pair loss ---
                real_global_labels = torch.ones(right_images.size(0))
                real_labels = Variable(real_global_labels).to(self.device)
                generator_loss += self.compute_g_loss(fake_pair_logit, real_labels)

                # -- compute image loss ---
                real_local_labels = torch.ones((right_images.size(0), 1, 5, 5))
                real_labels = Variable(real_local_labels).to(self.device)
                img_loss = self.compute_g_loss(fake_img_logit_local, real_labels)
                generator_loss += img_loss
                self.g_plot_dict[key].plot(to_numpy(img_loss).mean())

            if type(mean_var) == tuple:
                kl_loss = self.get_KL_Loss(mean_var[0], mean_var[1])
                kl_loss_val = to_numpy(kl_loss).mean()
                generator_loss += self.config["trainer"]["KL_coe"] * kl_loss
            else:
                # when trian 512HDGAN. KL loss is fixed since we assume 256HDGAN is trained.
                # mean_var actually returns pixel-wise l1 loss (see paper)
                generator_loss += mean_var

            self.kl_loss_plot.plot(kl_loss_val)
            generator_loss.backward()
            self.generator_optimizer.step()
            g_loss_val = to_numpy(generator_loss).mean()
            self.g_loss_plot.plot(g_loss_val)

            # train the discriminator
            self.generator.freeze()
            self.discriminator.unfreeze()
            self.discriminator_optimizer.zero_grad()

            noise = Variable(torch.randn(right_images['256'].size(0), self.noise_dim)).to(self.device)
            noise = noise.view(noise.size(0), self.noise_dim, 1, 1)
            fake_images, mean_var = self.to_img_dict_(self.generator(right_embeddings, noise))

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
                real_global_labels = torch.ones(right_images.size(0))
                fake_global_labels = -torch.ones(right_images.size(0))
                real_labels = Variable(real_global_labels).to(self.device)
                fake_labels = Variable(fake_global_labels).to(self.device)
                pair_loss = self.compute_d_pair_loss(real_logit, wrong_logit, fake_logit, real_labels, fake_labels)

                ''' compute disc image loss '''
                real_local_labels = torch.ones((right_images.size(0), 1, 5, 5))
                fake_local_labels = -torch.ones((right_images.size(0), 1, 5, 5))
                real_labels = Variable(real_local_labels).to(self.device)
                fake_labels = Variable(fake_local_labels).to(self.device)
                img_loss = self.compute_d_img_loss(wrong_img_logit_local, real_img_logit_local, fake_img_logit_local,
                                              real_labels, fake_labels)

                discriminator_loss += (pair_loss + img_loss)

                self.d_plot_dict[key].plot(to_numpy(img_loss).mean())

            discriminator_loss.backward()
            self.discriminator_optimizer.step()
            d_loss_val = to_numpy(discriminator_loss).mean()
            self.d_loss_plot.plot(d_loss_val)

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
        criterion = self.loss
        real_d_loss = criterion(real_logit, real_labels)
        wrong_d_loss = criterion(wrong_logit, fake_labels)
        fake_d_loss = criterion(fake_logit, fake_labels)

        discriminator_loss = real_d_loss + (wrong_d_loss + fake_d_loss) / 2.
        return discriminator_loss

    def compute_d_img_loss(self, wrong_img_logit, real_img_logit, fake_img_logit, real_labels, fake_labels):
        criterion = self.loss
        wrong_d_loss = criterion(wrong_img_logit, real_labels)
        real_d_loss = criterion(real_img_logit, real_labels)
        fake_d_loss = criterion(fake_img_logit, fake_labels)

        return fake_d_loss + (wrong_d_loss + real_d_loss) / 2

    def compute_g_loss(self, fake_logit, real_labels):
        criterion = self.loss
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