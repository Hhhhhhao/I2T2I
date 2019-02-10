import numpy as np
import torch
import yaml
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader.data_loaders import Text2ImageDataLoader
from model.gan_factory import gan_factory
from utils.utils_cls import Utils
from PIL import Image
import os


class Trainer(object):
    # gan_type can only be 'gan_cls' temporarily
    def __init__(self, gan_type, data_loader,
                 num_epochs, lr, save_path, l1_coef, l2_coef,
                 pre_trained_gen=None, pre_trained_disc=None):

        self.generator = torch.nn.DataParallel(gan_factory.generator_factory(gan_type).cuda())
        self.discriminator = torch.nn.DataParallel(gan_factory.discriminator_factory(gan_type).cuda())

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        self.gan_type = gan_type
        self.data_loader = data_loader

        # TODO: set noise dimension
        self.noise_dim = 100

        # TODO: set beta1 in Adam
        self.beta1 = 0.5

        self.num_epochs = num_epochs
        self.lr = lr

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path

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
        iteration = 0

        for epoch in range(self.num_epochs):
            for sample in self.data_loader:
                iteration += 1
                right_image = sample['right_image']
                right_embed = sample['right_embed']
                wrong_image = sample['wrong_image']

                right_image = Variable(right_image.float()).cuda()
                right_embed = Variable(right_embed.float()).cuda()
                wrong_image = Variable(wrong_image.float()).cuda()

                real_labels = torch.ones(right_image.size(0))
                fake_labels = torch.zeros(right_image.size(0))

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))


                real_labels = Variable(real_labels).cuda()
                smoothed_real_labels = Variable(smoothed_real_labels).cuda()
                fake_labels = Variable(fake_labels).cuda()



                # Train the discriminator
                self.discriminator.zero_grad()

                # real image, right text
                outputs, activation_real = self.discriminator(right_image, right_embed)
                real_loss = criterion(outputs, smoothed_real_labels)
                real_score = outputs

                # wrong image, right text
                outputs, _ = self.discriminator(wrong_image, right_embed)
                wrong_loss = criterion(outputs, fake_labels)
                wrong_score = outputs

                # fake image, right text
                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_image = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_image, right_embed)
                fake_loss = criterion(outputs, fake_labels)
                fake_score = outputs

                d_loss = real_loss + wrong_loss + fake_loss

                d_loss.backward()

                self.optimD.step()



                # Train the generator
                self.generator.zero_grad()
                noise = Variable(torch.randn(right_image.size(0), 100)).cuda()
                noise = noise.view(noise.size(0), 100, 1, 1)
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
                g_loss = criterion(outputs, real_labels) \
                         + self.l2_coef * l2_loss(activation_fake, activation_real.detach()) \
                         + self.l1_coef * l1_loss(fake_image, right_image)

                g_loss.backward()
                self.optimG.step()


            if epoch % 10 == 0:
                Utils.save_checkpoint(self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)

    def predict(self):
        for sample in self.data_loader:
            right_images = sample['right_images']
            right_embed = sample['right_embed']
            txt = sample['txt']

            if not os.path.exists('results/{0}'.format(self.save_path)):
                os.makedirs('results/{0}'.format(self.save_path))

            right_images = Variable(right_images.float()).cuda()
            right_embed = Variable(right_embed.float()).cuda()


            noise = Variable(torch.randn(right_images.size(0), 100)).cuda()
            noise = noise.view(noise.size(0), 100, 1, 1)
            fake_images = self.generator(right_embed, noise)


            for image, t in zip(fake_images, txt):
                im = Image.fromarray(image.data.mul_(127.5).add_(127.5).byte().permute(1, 2, 0).cpu().numpy())
                im.save('results/{0}/{1}.jpg'.format(self.save_path, t.replace("/", "")[:100]))
                print(t)
