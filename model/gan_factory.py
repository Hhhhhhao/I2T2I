from model import gan_cls
from model import lsgan_cls


class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan_cls':
            return gan_cls.generator()
        elif type =='lsgan_cls':
            return lsgan_cls.generator()
        elif type == 'wgan_cls':
            return lsgan_cls.generator()
        # elif type == 'vanilla_gan':
        #     return gan.generator()
        # elif type == 'vanilla_wgan':
        #     return wgan.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan_cls':
            return gan_cls.discriminator()
        elif type == 'lsgan_cls':
            return lsgan_cls.discriminator()
        elif type == 'wgan_cls':
            return lsgan_cls.discriminator()

        # elif type == 'vanilla_gan':
        #     return gan.discriminator()
        # elif type == 'vanilla_wgan':
        #     return wgan.discriminator()
