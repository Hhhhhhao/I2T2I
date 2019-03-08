from model import gan_cls
from model import lsgan_cls
from model import lsgan_sn_cls
from model import lsgan_mbd_cls


class gan_factory(object):
    @staticmethod
    def generator_factory(type):
        if type == 'gan_cls':
            return gan_cls.generator()
        elif type =='lsgan_cls':
            return lsgan_cls.generator()
        elif type == 'lsgan_sn_cls':
            return lsgan_sn_cls.generator()
        elif type == 'lsgan_cls_int':
            return lsgan_cls.generator()
        elif type == 'wgan_cls':
            return lsgan_cls.generator()
        elif type == 'lsgan_mbd_cls':
            return lsgan_mbd_cls.generator()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan_cls':
            return gan_cls.discriminator()
        elif type == 'lsgan_cls':
            return lsgan_cls.discriminator()
        elif type == 'lsgan_sn_cls':
            return lsgan_sn_cls.discriminator()
        elif type == 'lsgan_cls_int':
            return lsgan_cls.discriminator()
        elif type == 'wgan_cls':
            return lsgan_cls.discriminator()
        elif type == 'lsgan_mbd_cls':
            return lsgan_mbd_cls.discriminator()
