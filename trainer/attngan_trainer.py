import os
import torch
import numpy as np
from torch.autograd import Variable
from .base_trainer import BaseTrainer
from model import networks
from model.loss import attangan_discriminator_loss, attangan_generator_loss, KL_loss
from utils.util import convert_back_to_text
from collections import OrderedDict
dirname = os.path.dirname(__file__)


class AttnGANtrainer(BaseTrainer):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.
        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.
        Returns:
            the modified parser.
        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(exp_namem='AttnGAN', netG='synthesis', netD='synthesis')
        if is_train:
            parser.add_argument('--gamma1', type=float, default=4.0, help='gamma 1 for damsm')
            parser.add_argument('--gamma2', type=float, default=5.0, help='gamma 2 for damsm')
            parser.add_argument('--gamma3', type=float, default=10.0, help='gamma 3 for damsm')
            parser.add_argument('--g_lambda', type=float, default=5.0, help='gamma 3 for damsm')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(AttnGANtrainer, self).__init__(opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'D']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['fake_imgs', 'real_imgs']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G', 'D']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt=opt, gpu_ids=self.gpu_ids)
        self.netD = networks.define_D(opt=opt, gpu_ids=self.gpu_ids)
        self.rnn_encoder, self.cnn_encoder = networks.define_DAMSM(opt=opt, gpu_ids=self.gpu_ids)

        self.generator_loss = attangan_generator_loss
        self.discriminator_loss = attangan_discriminator_loss
        self.KL_loss = KL_loss

        # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.g_lr, betas=(opt.beta_1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_D = []
        for i in range(len(self.netD)):
            self.optimizer_D.append(torch.optim.Adam(self.netD[i].parameters(), lr=opt.g_lr, betas=(opt.beta_1, 0.999)))
            self.optimizers.append(self.optimizer_D[i])

        # setup noise
        self.noise = Variable(torch.FloatTensor(self.batch_size, 100), volatile=True)
        self.noise.to(self.device)

        # setup labels
        self.real_labels, self.fake_labels, self.match_labels = self.prepare_labels()

    def set_input(self, data):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.real_imgs = []
        self.real_imgs.append(data["right_images_64"].to(self.device))
        self.real_imgs.append(data["right_images_128"].to(self.device))
        self.real_imgs.append(data["right_images_256"].to(self.device))
        self.right_captions = data["right_captions"].to(self.device)
        self.right_caption_lengths = data["right_caption_lengths"].to(self.device)
        self.class_ids = np.array(data['class_id'])
        self.labels = torch.LongTensor(range(self.batch_size)).to(self.device)

        # # other image
        # self.wrong_images = []
        # self.wrong_images.append(data["wrong_images_64"].to(self.device))
        # self.wrong_images.append(data["wrong_images_128"].to(self.device))
        # self.wrong_images.append(data["wrong_images_256"].to(self.device))
        # self.wrong_captions = data["wrong_captions"].to(self.device)
        # self.wrong_caption_lengths = data["wrong_caption_lengths"].to(self.device)

    def prepare_labels(self):
        real_labels = Variable(torch.FloatTensor(self.batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(self.batch_size).fill_(0))
        match_labels = Variable(torch.LongTensor(range(self.batch_size)))
        if torch.cuda.is_available():
            real_labels = real_labels.cuda()
            fake_labels = fake_labels.cuda()
            match_labels = match_labels.cuda()

        return real_labels, fake_labels, match_labels

    def forward(self):
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef

        self.words_embs, self.sent_emb = self.rnn_encoder(self.right_captions, self.right_caption_lengths)
        self.words_embs, self.sent_emb = self.words_embs.detach(), self.sent_emb.detach()
        mask = (self.right_captions == 0)
        num_words = self.words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]

        #######################################################
        # (2) Generate fake images
        ######################################################
        self.noise.data.normal_(0, 1)
        self.fake_imgs, _, self.mu, self.logvar = self.netG(self.noise, self.sent_emb, self.words_embs, mask)

    def backward_D(self):
        """Calculate loss for the discriminator"""
        #######################################################
        # (3) calculate D network loss
        ######################################################
        self.loss_D = 0
        for i in range(len(self.netD)):
            self.netD[i].zero_grad()
            loss = self.discriminator_loss(self.netD[i], self.real_imgs[i], self.fake_imgs[i],
                                      self.sent_emb, self.real_labels, self.fake_labels)
            # backward and update parameters
            loss.backward()
            # optimizersD[i].step()
            self.loss_D += loss

    def backward_G(self):
        #######################################################
        # (4) Update G network: maximize log(D(G(z)))
        ######################################################
        # compute total loss for training G

        # do not need to compute gradient for Ds
        # self.set_requires_grad_value(netsD, False)
        self.netG.zero_grad()
        self.loss_G = self.generator_loss(self.netD, self.cnn_encoder, self.fake_imgs, self.real_labels,
                                           self.words_embs, self.sent_emb, self.match_labels,
                                           self.right_caption_lengths, self.class_ids, self.opt)
        kl_loss = self.KL_loss(self.mu, self.logvar)
        self.loss_G += kl_loss
        # backward and update parameters
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()   # compute the fake images from text embedding: G(s, w)
        # update D
        self.set_requires_grad(self.netD, True)

        # set D's gradients to zero
        for i in range(len(self.netD)):
            self.optimizer_D[i].zero_grad()
        self.backward_D()  # calculate gradients for D
        # update D's weights
        for i in range(len(self.netD)):
            self.optimizer_D[i].step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def get_current_visuals(self, vocab):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        wordidarray = self.right_captions.detach().cpu().numpy()
        for j, name in enumerate(self.visual_names):
            if isinstance(name, str):
                results = getattr(self, name)
                if type(results) is list:
                    for i, size in enumerate(['64', '128', '256']):
                        title = name + '-' + size
                        if i == 0 and j == 0 :
                            title = convert_back_to_text(wordidarray[0], vocab)
                        visual_ret[title] = results[i]
                else:
                    visual_ret[name] = results

        return visual_ret











