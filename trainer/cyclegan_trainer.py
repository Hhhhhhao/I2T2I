import os
import torch
import numpy as np
import itertools
from torch.autograd import Variable
from .base_trainer import BaseTrainer
from model import networks
from model.loss import KLLoss, AttnDiscriminatorLoss, AttnGeneratorLoss, CaptGANDiscriminatorLoss, CaptGANGeneratorLoss, SentLoss, WordLoss
from utils.util import convert_back_to_text, get_caption_lengths
from collections import OrderedDict
dirname = os.path.dirname(__file__)


class CycleGANTrainer(BaseTrainer):
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
        parser.set_defaults(exp_namem='CycleGAN')
        if is_train:
            parser.add_argument('--gamma1', type=float, default=4.0, help='gamma 1 for damsm')
            parser.add_argument('--gamma2', type=float, default=5.0, help='gamma 2 for damsm')
            parser.add_argument('--gamma3', type=float, default=10.0, help='gamma 3 for damsm')
            parser.add_argument('--g_lambda', type=float, default=5.0, help='gamma 3 for damsm')
            parser.add_argument('--lambda_I', type=float, default=10.0, help='gamma 1 for damsm')
            parser.add_argument('--lambda_S', type=float, default=10.0, help='gamma 2 for damsm')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super(CycleGANTrainer, self).__init__(opt)
        self.lambda_I = opt.lambda_I
        self.lambda_S = opt.lambda_S
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_S', 'D_S', 'Cycle_S', 'G_I', 'D_I', 'Cycle_I']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_imgs', 'fake_imgs', 'rec_images']
        self.caption_names = ['real_captions', 'fake_captions', 'rec_captions']

        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G_S', 'D_S', 'G_I', 'D_I']
        # define networks (both generator and discriminator) for CaptGAN
        opt.netG = 'caption'
        opt.netD = 'caption'
        self.netG_S = networks.define_G(opt=opt, gpu_ids=self.gpu_ids)
        self.netD_S = networks.define_D(opt=opt, gpu_ids=self.gpu_ids)

        # define networks (both generator and discriminator) for AttnGAN
        opt.netG = 'synthesis'
        opt.netD = 'synthesis'
        self.netG_I = networks.define_G(opt=opt, gpu_ids=self.gpu_ids)
        self.netD_I = networks.define_D(opt=opt, gpu_ids=self.gpu_ids)

        # define networks for
        self.rnn_encoder, self.cnn_encoder = networks.define_DAMSM(opt=opt, gpu_ids=self.gpu_ids)

        # define loss functions
        self.caption_generator_loss = CaptGANGeneratorLoss()
        self.caption_discriminator_loss = CaptGANDiscriminatorLoss()
        self.synthesis_generator_loss = AttnGeneratorLoss(opt)
        self.synthesis_discriminator_loss = AttnDiscriminatorLoss()
        self.synthesis_kl_loss = KLLoss()
        self.cycle_consistency_loss = torch.nn.L1Loss()

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_S.parameters(), self.netG_I.parameters()), lr=opt.g_lr, betas=(opt.beta_1, 0.999))
        self.optimizers.append(self.optimizer_G)
        self.optimizer_D = torch.optim.Adam(
            itertools.chain(self.netD_S.parameters(), self.netD_I[0].parameters(), self.netD_I[1].parameters(), self.netD_I[2].parameters()),
            lr=opt.d_lr, betas=(opt.beta_1, 0.999))
        self.optimizers.append(self.optimizer_D)

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
        self.real_captions = data["right_captions"].to(self.device)
        self.real_caption_lengths = data["right_caption_lengths"].to(self.device)
        self.class_ids = np.array(data['class_id'])
        self.labels = torch.LongTensor(range(self.batch_size)).to(self.device)

        # other image
        self.wrong_images = []
        self.wrong_images.append(data["wrong_images_64"].to(self.device))
        self.wrong_images.append(data["wrong_images_128"].to(self.device))
        self.wrong_images.append(data["wrong_images_256"].to(self.device))
        self.wrong_captions = data["wrong_captions"].to(self.device)
        self.wrong_caption_lengths = data["wrong_caption_lengths"].to(self.device)

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

        # Forward from sentence to image G(S) = I
        self.fake_imgs, self.mu, self.logvar, self.real_words_embs, self.real_sent_emb = self.forward_G_I(self.real_captions, self.real_caption_lengths)

        # Reconstruction from fake image to rec sentence F(G(S)) = S_hat
        _, _, self.rec_captions, self.rec_caption_lengths = self.forward_G_S(self.fake_imgs[-1])

        # Forward from image to sentence F(I) = S
        self.rewards, self.props, self.fake_captions, self.fake_caption_lengths = self.forward_G_S(self.real_imgs[-1])

        # Reconstruction from fake sentence to rec images G(F(I)) = I_hat
        self.rec_images, _, _ , _, _ = self.forward_G_I(self.fake_captions, self.fake_caption_lengths)


        ###### Compute images feauters
        _, self.rec_sent_emb = self.rnn_encoder(self.rec_captions, self.rec_caption_lengths)
        _, self.rec_image_emb = self.cnn_encoder(self.rec_images[-1])
        _, self.real_image_emb = self.cnn_encoder(self.real_imgs[-1])


    def forward_G_S(self, images):

        """ Forward through the generator of CaptGAN"""
        ####### Forward CaptGAN Generator #########
        # feature forward to get genrated captions for training the discriminator
        features = self.netG_S.feature_forward(images)
        fake_captions = self.netG_S.feature_to_text(features)
        fake_captions, fake_caption_lengths = get_caption_lengths(fake_captions)
        fake_captions = fake_captions.detach()
        fake_captions.to(self.device)
        fake_caption_lengths.to(self.device)

        # reward forward for training CaptGAN generator
        rewards, props = self.netG_S.reward_forward(images, self.netD_S, monte_carlo_count=18)

        return rewards, props, fake_captions, fake_caption_lengths

    def forward_G_I(self, captions, caption_lengths):

        """ Forward through the generator of AttnGAN """
        ####### Forward AttnGAN Generator #########
        # words_embs: batch_size x nef x seq_len
        # sent_emb: batch_size x nef
        words_embs, sent_emb = self.rnn_encoder(captions, caption_lengths)
        words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
        mask = (captions == 0)
        num_words = words_embs.size(2)
        if mask.size(1) > num_words:
            mask = mask[:, :num_words]
        self.noise.data.normal_(0, 1)
        fake_imgs, _, mu, logvar = self.netG_I(self.noise, sent_emb, words_embs, mask)

        return fake_imgs, mu, logvar, words_embs, sent_emb

    def backward_D_S(self):
        """Calculate loss for the discriminator of CaptGAN"""
        self.netD_S.zero_grad()
        evaluator_scores = self.netD_S(self.real_imgs[-1], self.real_captions, self.real_caption_lengths)
        generator_scores = self.netD_S(self.real_imgs[-1], self.fake_captions, self.fake_caption_lengths)
        other_scores = self.netD_S(self.real_imgs[-1], self.wrong_captions, self.wrong_caption_lengths)
        batch_size = evaluator_scores.size(0)
        self.loss_D_S = self.caption_discriminator_loss(evaluator_scores.view(batch_size),
                                                        generator_scores.view(batch_size),
                                                        other_scores.view(batch_size),
                                                        self.real_labels,
                                                        self.fake_labels)
        self.loss_D_S.backward()

    def backward_D_I(self):
        """Calculate loss for the discriminator of AttnGAN"""
        self.loss_D_I = 0
        for i in range(len(self.netD_I)):
            self.netD_I[i].zero_grad()
            loss = self.synthesis_discriminator_loss(self.netD_I[i], self.real_imgs[i], self.fake_imgs[i],
                                      self.real_sent_emb, self.real_labels, self.fake_labels)
            # backward and update parameters
            loss.backward()
            # optimizersD[i].step()
            self.loss_D_I += loss

    def backward_G(self):
        """Calculate the loss for generators G and F"""
        # compute total loss for training attention generator for synthesis #
        self.netG_I.zero_grad()
        self.loss_G_I = self.synthesis_generator_loss(self.netD_I, self.cnn_encoder, self.fake_imgs, self.real_labels,
                                           self.real_words_embs, self.real_sent_emb, self.match_labels,
                                           self.real_caption_lengths, self.class_ids)
        kl_loss = self.synthesis_kl_loss(self.mu, self.logvar)
        self.loss_G_I += kl_loss

        # compute loss for training caption generator using Policy Gradient #
        self.loss_G_S = self.caption_generator_loss(self.rewards, self.props)

        # compute perceptually cycle consistency loss using DAMSM
        self.loss_cycle_S = self.cycle_consistency_loss(self.real_sent_emb, self.rec_sent_emb) * self.lambda_S

        self.loss_cycle_I = self.cycle_consistency_loss(self.real_image_emb, self.rec_sent_emb) * self.lambda_I

        # backward and update parameters
        self.loss_G = self.loss_G_I + self.loss_G_S + self.loss_cycle_I + self.loss_cycle_S
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # update G
        self.set_requires_grad(self.netD_I + [self.netD_S], False)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

        # update D
        self.set_requires_grad(self.netD_I + [self.netD_S], True)
        # set D's gradients to zero
        self.optimizer_D.zero_grad()
        self.backward_D_I()  # calculate gradients for D_I
        self.backward_D_S()  # calculate gradients for D_S
        self.optimizer_D.step()

    def get_current_visuals(self, vocab):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()

        for vis_name, cap_name in zip(self.visual_names, self.caption_names):
            if isinstance(vis_name, str):
                results = getattr(self, vis_name)
                results = results[-1]
                visual_ret[vis_name] = results

        return visual_ret











