import argparse
import os
from utils import util
import torch
import datetime
import model


class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='data/', help='path to images')
        parser.add_argument('--exp_name', type=str, default='CycleGAN', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./saved', help='models are saved here')
        parser.add_argument('--n_gpu', type=int, default=8, help='number of gpu')

        # model parameters
        parser.add_argument('--model', type=str, default='cyclegan', help='chooses which model to use. [attngan | captiongan | cyclegan]')


        # parameters for attnGAN generator
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--text_embedding_dim', type=int, default=256, help='text embedding dimension of damsm')
        parser.add_argument('--condition_dim', type=int, default=128, help='condition augmentation dimension')
        parser.add_argument('--noise_dim', type=int, default=100, help='noise dim for generator')
        parser.add_argument('--branch_num', type=int, default=3, help='generate what size images [1 for 64 | 2 for 128 | 3 for 256]')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        # parser.add_argument('--netD_I', type=str, default='synthesis', help='specify discriminator architecture [caption | synthesis].')
        # parser.add_argument('--netG_I', type=str, default='synthesis', help='specify generator architecture [caption | synthesis]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')


        # parameters for image Caption GAN
        # parser.add_argument('--netD_S', type=str, default='caption', help='specify discriminator architecture [caption | synthesis].')
        # parser.add_argument('--netG_S', type=str, default='caption', help='specify generator architecture [caption | synthesis]')
        parser.add_argument('--image_embedding_dim', type=int, default=256, help='image feature dimension for CaptGAN')


        # dataset parameters
        parser.add_argument('--dataset_name', type=str, default='flowers', help='chooses which datasets to loaded. [ birds | flowers | CoCo]')
        parser.add_argument('--which_set', type=str, default='train', help='chooses which set of data to use [train | valid | test]')
        parser.add_argument('--image_size', type=int, default=256, help='final image size to load')
        parser.add_argument('--num_workers', default=0, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        parser.add_argument('--validation_split', type=float, default=0.02, help='validation split of COCO')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = model.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # # modify dataset-related parser options
        # dataset_name = opt.dataset_name
        # dataset_option_setter = data_loader.get_option_setter(dataset_name)
        # parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        # setup directory for checkpoint saving
        # start_time = datetime.datetime.now().strftime('%m%d_%H%M%S')
        save_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)
        opt.save_dir = save_dir
        util.mkdirs(save_dir)
        file_name = os.path.join(save_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)
        self.opt = opt
        return self.opt