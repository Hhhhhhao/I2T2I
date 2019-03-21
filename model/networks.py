import os
import torch
from torch.nn import init
from torch.optim import lr_scheduler

from model.attngan_modules import D_NET64, D_NET128, D_NET256, G_NET
from model.damsm_modules import DAMSM_RNN_Encoder, DAMSM_CNN_Encoder
from model.captiongan_modules import ConditionalGenerator, Evaluator

dirname = os.path.dirname(__file__)
main_dirname = os.path.dirname(dirname)
birds_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-Birds/0315_235622/model_best.pth')
flowers_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-Flowers/0316_120632/model_best.pth')
coco_damsm = os.path.join(main_dirname, 'output/Deep-Attentional-Multimodal-Similarity-CoCo/0313_161600/model_best.pth')


###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine
    For 'linear', we keep the same learning rate for the first <opt.epochs> epochs
    and linearly decay the rate to zero over the next <opt.nepoch_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.epochs) / float(opt.nepoch_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


###############################################################################
# Generator Define Function & Discriminator Define Function
###############################################################################
def define_G(opt, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator
    Parameters:
        netG (str) -- the architecture's name: caption | synthesis
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator
    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597
        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).
    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None

    if opt.netG == 'caption':
        net = ConditionalGenerator(
                 image_embed_size=opt.image_embedding_dim,
                 word_embed_size=opt.image_embedding_dim,
                 noise_dim=opt.noise_dim,
                 vocab_size=opt.vocab_size)
    elif opt.netG == 'synthesis':
        net = G_NET(opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)

    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(opt, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        netD (str)         -- the architecture's name: caption | synthesis
        opt  (parser option)
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    """
    net = None

    if opt.netD == 'caption':
        net = Evaluator(
                 word_embed_size=opt.image_embedding_dim,
                 sentence_embed_size=opt.image_embedding_dim,
                 vocab_size=opt.vocab_size,)
        return init_net(net, init_type, init_gain, gpu_ids)
    elif opt.netD == 'synthesis':  # more options
        net = []
        net_64 = D_NET64(opt)
        net_128 = D_NET128(opt)
        net_256 = D_NET256(opt)
        net.append(init_net(net_64, init_type, init_gain, gpu_ids))
        net.append(init_net(net_128, init_type, init_gain, gpu_ids))
        net.append(init_net(net_256, init_type, init_gain, gpu_ids))
        return net
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)


def define_DAMSM(opt, gpu_ids=[]):
    """
    Create Pre-trained DAMSM rnn encoder and cnn encoder
    :param dataset_name (str) -- the dataset name: birds | flowers | CoCo
    :param vocab_size (int) -- vocabulary size of the dataset used:
    :param gpu_ids (list)-- which GPUs the network runs on: e.g., 0,1,2
    :return: a pretrained rnn encoder and cnn encoder
    """

    rnn_encoder = DAMSM_RNN_Encoder(
        vocab_size=opt.vocab_size,
        word_embed_size=256,
        lstm_hidden_size=256
    )
    cnn_encoder = DAMSM_CNN_Encoder(embedding_size=256)

    if len(gpu_ids) > 1:
        assert(torch.cuda.is_available())
        device = torch.device('cuda:0')
        rnn_encoder.to(gpu_ids[0])
        cnn_encoder.to(gpu_ids[0])
        rnn_encoder = torch.nn.DataParallel(rnn_encoder, gpu_ids)  # multi-GPUs
        cnn_encoder = torch.nn.DataParallel(cnn_encoder, gpu_ids)
    else:
        # # TODO add here for test on computer
        # rnn_encoder = torch.nn.DataParallel(rnn_encoder, gpu_ids)  # multi-GPUs
        # cnn_encoder = torch.nn.DataParallel(cnn_encoder, gpu_ids)
        device = torch.device('cpu')
        rnn_encoder.to(device)
        cnn_encoder.to(device)

    if "d" in opt.dataset_name:  # birds
        resume_path = birds_damsm
    elif "r" in opt.dataset_name: # flowers
        resume_path = flowers_damsm
    elif "C" in opt.dataset_name: # coco
        resume_path = coco_damsm
    else:
        raise ValueError("cannot find corresponding damsm model path")
    checkpoint = torch.load(resume_path, map_location=device)

    rnn_encoder.load_state_dict(checkpoint["rnn_state_dict"])
    for p in rnn_encoder.parameters():
        p.requires_grad = False

    cnn_encoder.load_state_dict(checkpoint["cnn_state_dict"])
    for p in cnn_encoder.parameters():
        p.requires_grad = False

    return rnn_encoder, cnn_encoder

