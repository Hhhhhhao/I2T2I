import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import logging
from model.gan_factory import gan_factory
from utils.utils_cls import Utils
from PIL import Image
import os


class Trainer(object):
    # gan_type can only be 'gan_cls' temporarily
    def __init__(self, gan_type, train_data_loader, valid_data_loader,
                 num_epochs, lr, save_path, l1_coef, l2_coef,
                 pre_trained_gen=None, pre_trained_disc=None):

        self.logger = logging.getLogger(self.__class__.__name__)

        # setup GPU device if available, move model into configured device
        if torch.cuda.is_available():
            print("use GPU")
            self.logger.info('use GPU')
            self.device = torch.device('cuda')
        else:
            print("use CPU")
            self.logger.info('use CPU')
            self.device = torch.device('cpu')

        # self.model = model.to(self.device)

        self.generator = torch.nn.DataParallel(gan_factory.generator_factory(gan_type).to(self.device))
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(gan_type).to(self.device))

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc, map_location='cpu'))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen, map_location='cpu'))
        else:
            self.generator.apply(Utils.weights_init)

        self.gan_type = gan_type
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.noise_dim = 100

        self.beta1 = 0.5

        self.num_epochs = num_epochs
        self.lr = lr

        # did not use it here
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

        if train_data_loader:
            self.log_step = int(np.sqrt(self.train_data_loader.batch_size))

    def train(self):

        if self.gan_type == 'gan_cls':
            self._train_gan_cls()
        # elif self.gan_type == 'vanilla_wgan':
        #     self._train_vanilla_wgan()
        # elif self.gan_type == 'vanilla_gan':
        #     self._train_vanilla_gan()

    def _train_gan_cls(self):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        for epoch in range(self.num_epochs):

            for batch_idx, sample in enumerate(self.train_data_loader):
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                wrong_images = sample['wrong_images']

                right_images = Variable(right_images.float()).to(self.device)
                right_embed = Variable(right_embed.float()).to(self.device)
                wrong_images = Variable(wrong_images.float()).to(self.device)

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = Variable(real_labels).to(self.device)
                smoothed_real_labels = Variable(smoothed_real_labels).to(self.device)
                fake_labels = Variable(fake_labels).to(self.device)

                # Train the discriminator
                self.discriminator.zero_grad()

                # real image, right text
                outputs, activation_real = self.discriminator(right_images, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                # wrong image, right text
                outputs, _ = self.discriminator(wrong_images, right_embed)
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                # fake image, right text
                noise = Variable(torch.randn(right_images.size(0), self.noise_dim)).to(self.device)
                noise = noise.view(noise.size(0), self.noise_dim, 1, 1)
                fake_image = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_image, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + (wrong_loss + fake_loss) / 2.0

                d_loss.backward()

                self.optimD.step()

                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), self.noise_dim)).to(self.device)
                noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                fake_image = self.generator(right_embed, noise)
                outputs, activation_fake = self.discriminator(fake_image, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # image statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real image, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                # ===========================================
                g_loss = criterion(outputs, real_labels)
                # + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                # + self.l1_coef * l1_loss(fake_image, right_images)

                g_loss.backward()
                self.optimG.step()

                # log
                if batch_idx % self.log_step == 0:
                    self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] G_Loss: {:.6f} D_Loss: {:.6f}'.format(
                        epoch,
                        batch_idx * self.train_data_loader.batch_size,
                        len(self.train_data_loader.dataset),
                        100.0 * batch_idx / len(self.train_data_loader),
                        g_loss.item(),
                        d_loss.item()))

            if epoch % 5 == 0:
                self.logger.info("save checkpoints")
                Utils.save_checkpoint(self.discriminator, self.generator, self.save_path, self.checkpoints_path, epoch)

            if epoch % 10 == 0:
                self.logger.info("predict first batch for valid")
                self.predict(data_loader=self.valid_data_loader, valid=True, epoch=epoch)

    def predict(self, data_loader, valid=False, epoch=None):

        if valid:

            for batch_idx, sample in enumerate(data_loader):
                if batch_idx != 0:
                    break
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                txt = sample['txt']

                if not os.path.exists('{0}/results/epoch_{1}'.format(self.save_path, epoch)):
                    os.makedirs('{0}/results/epoch_{1}'.format(self.save_path, epoch))

                right_images = Variable(right_images.float()).to(self.device)
                right_embed = Variable(right_embed.float()).to(self.device)

                noise = Variable(torch.randn(right_images.size(0), self.noise_dim)).to(self.device)
                noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                fake_images = self.generator(right_embed, noise)

                for image, t in zip(fake_images, txt):
                    im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('{0}/results/epoch_{1}/{2}.jpg'.format(self.save_path, epoch, t.replace("/", "")[:100]))

        else:

            for sample in data_loader:
                right_images = sample['right_images']
                right_embed = sample['right_embed']
                txt = sample['txt']

                if not os.path.exists('{0}/results/test'.format(self.save_path)):
                    os.makedirs('{0}/results/test'.format(self.save_path))

                right_images = Variable(right_images.float()).to(self.device)
                right_embed = Variable(right_embed.float()).to(self.device)

                noise = Variable(torch.randn(right_images.size(0), self.noise_dim)).to(self.device)
                noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

                fake_images = self.generator(right_embed, noise)

                for image, t in zip(fake_images, txt):
                    im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save('{0}/results/test/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                    print(t)
