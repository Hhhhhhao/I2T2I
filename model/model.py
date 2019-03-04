import torch
import torch.nn as nn
import functools
from torchgan.layers import ResidualBlock2d

from base import BaseModel
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_block(input_dim, output_dim, kernel_size=3, stride=1):
    seq = []
    if kernel_size != 1:
        seq += [nn.ReflectionPad2d(1)]

    seq += [nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride, padding=0, bias=True),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(0.2)]

    return nn.Sequential(*seq)


def branch_out(in_dim, out_dim=3):
    _layers = [ nn.ReflectionPad2d(1),
                nn.Conv2d(in_dim, out_dim,
                kernel_size=3, padding=0, bias=False)]
    _layers += [nn.Tanh()]

    return nn.Sequential(*_layers)


## Weights init function, DCGAN use 0.02 std
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)


class CAEmbedding(BaseModel):
    def __init__(self, text_dim, embed_dim):
        super(CAEmbedding, self).__init__()

        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.linear = nn.Linear(self.text_dim, self.embed_dim*2, bias=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def encode(self, text_embedding):
        x = self.relu(self.linear(text_embedding))
        mean = x[:, :self.embed_dim]
        log_var = x[:, self.embed_dim:]
        return mean, log_var

    def reparametrize(self, mean, log_var):
        std = log_var.mul(0.5).exp_()

        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()

        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, text_embedding):
        mean, log_var = self.encode(text_embedding)
        c_code = self.reparametrize(mean, log_var)
        return c_code, mean, log_var


class Sent2FeatMap(nn.Module):
    # used to project a sentence code into a set of feature maps
    def __init__(self, in_dim, row, col, channel, activ=None):
        super(Sent2FeatMap, self).__init__()
        self.__dict__.update(locals())
        out_dim = row*col*channel
        norm_layer = functools.partial(nn.BatchNorm1d, affine=True)
        _layers = [nn.Linear(in_dim, out_dim)]
        _layers += [norm_layer(out_dim)]
        if activ is not None:
            _layers += [activ]
        self.out = nn.Sequential(*_layers)

    def forward(self, inputs):
        output = self.out(inputs)
        output = output.view(-1, self.channel, self.row, self.col)
        return output


class ResidualBlock(BaseModel):
    def __init__(self, input_dim):
        super(ResidualBlock, self).__init__()
        self.res_block = ResidualBlock2d(
            filters=[input_dim, input_dim, input_dim],
            kernels=[3, 3],
            strides=[1, 1],
            paddings=[1, 1]
        )

    def forward(self, input):
        return self.res_block(input)


class ImageDownSample(BaseModel):
    def __init__(self, input_size, num_chan, out_dim):
        """
            Parameters:
            ----------
            input_size: int
                input image size, can be 64, or 128, or 256
            num_chan: int
                channel of input images.
            out_dim : int
                the channel dimension of generated image code.
        """

        super(ImageDownSample, self).__init__()
        self.__dict__.update(locals())

        _layers = []
        # use large kernel_size at the end to prevent using zero-padding and stride
        if input_size == 64:
            curr_dim = 128
            _layers += [conv_block(num_chan, curr_dim, kernel_size=3, stride=2)]  # 32
            _layers += [conv_block(curr_dim, curr_dim*2, kernel_size=3, stride=2)]  # 16
            _layers += [conv_block(curr_dim*2, curr_dim*4, kernel_size=3, stride=2)]  # 8
            _layers += [conv_block(curr_dim*4, out_dim, kernel_size=3, stride=1)] # 8

        if input_size == 128:
            curr_dim = 64
            _layers += [conv_block(num_chan, curr_dim, kernel_size=3, stride=2)]  # 64
            _layers += [conv_block(curr_dim, curr_dim*2, kernel_size=3, stride=2)]  # 32
            _layers += [conv_block(curr_dim*2, curr_dim*4, kernel_size=3, stride=2)]  # 16
            _layers += [conv_block(curr_dim*4, curr_dim*8, kernel_size=3, stride=2)] # 8
            _layers += [conv_block(curr_dim*8, out_dim, kernel_size=3, stride=1)] # 8

        if input_size == 256:
            curr_dim = 32 # for testing
            _layers += [conv_block(num_chan, curr_dim, kernel_size=3, stride=2)]  # 128
            _layers += [conv_block(curr_dim, curr_dim*2, kernel_size=3, stride=2)]  # 64
            _layers += [conv_block(curr_dim*2, curr_dim*4, kernel_size=3, stride=2)]  # 32
            _layers += [conv_block(curr_dim*4, curr_dim*8, kernel_size=3, stride=2)] # 16
            _layers += [conv_block(curr_dim*8, out_dim, kernel_size=3, stride=2)] # 8

        self.encode = nn.Sequential(*_layers)

    def forward(self, inputs):

        out = self.encode(inputs)
        return out


class DiscClassifier(BaseModel):
    def __init__(self, enc_dim, emb_dim, kernel_size):
        """
           Parameters:
           ----------
           enc_dim: int
               the channel of image code.
           emb_dim: int
               the channel of sentence code.
           kernel_size : int
               kernel size used for final convolution.
       """

        super(DiscClassifier, self).__init__()
        self.__dict__.update(locals())

        inp_dim = enc_dim + emb_dim

        _layers = [conv_block(inp_dim, enc_dim, kernel_size=1, stride=1),
                   nn.Conv2d(enc_dim, 1, kernel_size=kernel_size, padding=0, bias=True)]

        self.node = nn.Sequential(*_layers)

    def forward(self, sent_code, img_code):
        sent_code = sent_code.unsqueeze(-1).unsqueeze(-1)
        dst_shape = list(sent_code.size())
        dst_shape[1] = sent_code.size()[1]
        dst_shape[2] = img_code.size()[2]
        dst_shape[3] = img_code.size()[3]
        sent_code = sent_code.expand(dst_shape)
        comp_inp = torch.cat([img_code, sent_code], dim=1)
        output = self.node(comp_inp)
        chn = output.size()[1]
        output = output.view(-1, chn)

        return output


class HDGANGenerator(BaseModel):

    def __init__(self,
                 text_embed_dim=1024,
                 ca_code_dim=128,
                 noise_dim=128,
                 num_resblock=1,
                 side_output_at=[64, 128, 256]):
        super(HDGANGenerator, self).__init__()
        self.__dict__.update(locals())
        # feature map dimension reduce at which resolution
        reduce_dim_at = [8, 32, 128, 256]
        # different sacles for all network
        num_scales = [4, 8, 16, 32, 64, 128, 256]
        # initialize feature map dimension
        curr_dim = 1024

        self.sent2featmap = Sent2FeatMap(ca_code_dim+noise_dim, 4, 4, curr_dim)
        self.side_output_at = side_output_at
        self.ca_embedding = CAEmbedding(text_embed_dim, ca_code_dim)

        for i in range(len(num_scales)):
            seq = []
            # upsampling
            if i != 0:
                seq += [nn.Upsample(scale_factor=2, mode='nearest')]

            # if need to reduce dimension
            if num_scales[i] in reduce_dim_at:
                seq += [conv_block(curr_dim, curr_dim//2, kernel_size=3)]
                curr_dim = curr_dim//2
            # add residual blocks
            for n in range(num_resblock):
                seq += [ResidualBlock(curr_dim)]

            # add main convolutional module
            setattr(self, 'scale_%d' % (num_scales[i]), nn.Sequential(*seq))

            if num_scales[i] in self.side_output_at:
                setattr(self, 'tensor_to_img_%d' %
                        (num_scales[i]), branch_out(curr_dim))

        self.apply(weights_init)
        print('>> Init HDGAN Generator')
        print('\t side output at {}'.format(str(side_output_at)))

    def forward(self, text_embedding, noise):
        """

        :param text_embedding: [batch_size, sent_dim], sentence emcodeing using DAMSM or Char-rnn
        :param noise: [batch_size, noise_dim] noise_input
        :return:
        """

        text_random, mean, logsigma = self.ca_embedding(text_embedding)
        x_in = torch.cat([text_random, noise], dim=1)

        # transform input vector into feature maps 4x4x1024
        x = self.sent2featmap(x_in)

        x_4 = self.scale_4(x)
        x_8 = self.scale_8(x_4)
        x_16 = self.scale_16(x_8)
        x_32 = self.scale_32(x_16)

        # skip 4x4 feature map to 32 and send to 64
        x_64 = self.scale_64(x_32)
        output_64 = self.tensor_to_img_64(x_64)

        # skip 8x8 feature map to 64 and send to 128
        x_128 = self.scale_128(x_64)
        output_128 = self.tensor_to_img_128(x_128)

        # skip 16x16 feature map to 128 and send to 256
        x_256 = self.scale_256(x_128)
        output_256 = self.tensor_to_img_256(x_256)

        return output_64, output_128, output_256, mean, logsigma

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class HDGANDiscriminator(BaseModel):
    def __init__(self,
                 text_embed_dim=1024,
                 ca_code_dim=128,
                 side_output_at=[64, 128, 256]):
        super(HDGANDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.num_chan = 3
        self.enc_dim = 128 * 4
        activation = nn.LeakyReLU(0.2)
        bn = functools.partial(nn.BatchNorm2d, affine=True)

        if 64 in side_output_at:
            self.img_encoder_64 = ImageDownSample(64, self.num_chan, self.enc_dim)  # enc_dim x 4 x 4
            self.pair_disc_64 = DiscClassifier(self.enc_dim, ca_code_dim, kernel_size=4)  # scalar
            self.local_img_disc_64 = nn.Conv2d(self.enc_dim, 1, kernel_size=4, padding=0, bias=True)  # 1 x 1 x 1
            _layers = [nn.Linear(text_embed_dim, ca_code_dim), nn.LeakyReLU(0.2)]
            self.context_emb_pipe_64 = nn.Sequential(*_layers)

        if 128 in side_output_at:
            self.img_encoder_128 = ImageDownSample(128, self.num_chan, self.enc_dim)  # enc_dim x 4 x 4
            self.pair_disc_128 = DiscClassifier(self.enc_dim, ca_code_dim, kernel_size=4)
            self.local_img_disc_128 = nn.Conv2d(self.enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(text_embed_dim, ca_code_dim), nn.LeakyReLU(0.2)]
            self.context_emb_pipe_128 = nn.Sequential(*_layers)

        if 256 in side_output_at:
            self.img_encoder_256 = ImageDownSample(256, self.num_chan, self.enc_dim)  # 8 x 8
            self.pair_disc_256 = DiscClassifier(self.enc_dim, ca_code_dim, kernel_size=4)
            self.pre_encode = conv_block(self.enc_dim, self.enc_dim, kernel_size=3, stride=1)
            self.local_img_disc_256 = nn.Conv2d(self.enc_dim, 1, kernel_size=4, padding=0, bias=True)
            _layers = [nn.Linear(text_embed_dim, ca_code_dim), nn.LeakyReLU(0.2)]
            self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)
        print('>> Init HDGAN Discriminator')
        print('\t Add adversarial loss at scale {}'.format(str(side_output_at)))

    def forward(self, images, text_embeddings):
        """

        :param images: (batch_size, channels, h, w) input image tensor
        :param text_embeddings: (batch_size, text_embed_size) corresponding embedding
        :return:
        """
        img_size = images.size()[3]
        assert img_size in [64, 128, 256], "wrong input size {} in discriminator".format(img_size)

        img_encoder = getattr(self, 'img_encoder_{}'.format(img_size))
        local_img_disc = getattr(self, 'local_img_disc_{}'.format(img_size))
        pair_disc = getattr(self, 'pair_disc_{}'.format(img_size))
        context_emb_pipe = getattr(self, 'context_emb_pipe_{}'.format(img_size))

        text_code = context_emb_pipe(text_embeddings)
        img_code = img_encoder(images)

        if img_size == 256:
            pre_img_code = self.pre_encode(img_code)
            pair_disc_out = pair_disc(text_code, pre_img_code)
        else:
            pair_disc_out = pair_disc(text_code, img_code)

        local_img_disc_out = local_img_disc(img_code)

        return pair_disc_out, local_img_disc_out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class DeepSupervisedHDGANDiscriminator(BaseModel):
    def __init__(self,
                 text_embed_dim=1024,
                 ca_code_dim=128,
                 side_output_at=[64, 128, 256]):
        super(DeepSupervisedHDGANDiscriminator, self).__init__()
        self.__dict__.update(locals())

        self.num_chan = 3
        self.enc_dim = 128 * 4
        activation = nn.LeakyReLU(0.2)
        bn = functools.partial(nn.BatchNorm2d, affine=True)


        self.convs = {}
        curr_dim = 32  # for testing
        self.convs['128_down'] = conv_block(self.num_chan, curr_dim, kernel_size=3, stride=2) # 32 x 128 x 128
        self.convs['128_fp'] = conv_block(self.num_chan, curr_dim, kernel_size=3, stride=1)  # 32 x 128 x 128
        self.convs['128_conv'] = conv_block(curr_dim*2, curr_dim, kernel_size=3, stride=2) # 32 x 128 x 128

        self.convs['64_down'] = conv_block(curr_dim, curr_dim * 2, kernel_size=3, stride=2)  # 64 x 64 x 64
        self.convs['64_fp'] = conv_block(self.num_chan, curr_dim * 2, kernel_size=3, stride=2)  # 64 x 64 x 64
        self.convs['64_conv'] = conv_block(curr_dim * 2, curr_dim * 2, kernel_size=3, stride=2)  # 64 x 64 x 64

        self.convs['32_down'] = conv_block(curr_dim * 2, curr_dim * 4, kernel_size=3, stride=2) # 128 x 32 x 32
        self.convs['16_down'] = conv_block(curr_dim * 4, curr_dim * 8, kernel_size=3, stride=2)  # 256 x 16 x 16
        self.convs['8_down'] = conv_block(curr_dim * 8, curr_dim * 16, kernel_size=3, stride=2)  # 8

        self.pair_disc_256 = DiscClassifier(self.enc_dim, ca_code_dim, kernel_size=4)
        self.pre_encode = conv_block(self.enc_dim, self.enc_dim, kernel_size=3, stride=1)
        self.local_img_disc_256 = nn.Conv2d(self.enc_dim, 1, kernel_size=4, padding=0, bias=True)
        _layers = [nn.Linear(text_embed_dim, ca_code_dim), nn.LeakyReLU(0.2)]
        self.context_emb_pipe_256 = nn.Sequential(*_layers)

        self.apply(weights_init)
        print('>> Init HDGAN Discriminator')
        print('\t Add adversarial loss at scale {}'.format(str(side_output_at)))

    def forward(self, feature_maps_64, feature_maps_128, images_256, text_embeddings):
        """

        :param images: (batch_size, channels, h, w) input image tensor
        :param text_embeddings: (batch_size, text_embed_size) corresponding embedding
        :return:
        """
        assert feature_maps_64.size()[3] == 64, "wrong feature map size {} in discriminator".format(feature_maps_64.size())
        assert feature_maps_128.size()[3] == 128, "wrong feature map size {} in discriminator".format(
            feature_maps_128.size())
        assert images_256.size()[3] == 256, "wrong image size {} in discriminator".format(
            images_256.size())

        text_code = self.context_emb_pipe_256(text_embeddings)
        images_128 = self.convs['128_down'](images_256)
        feature_maps_128 = self.convs['128_fp'](feature_maps_128)
        x_128 = torch.cat([images_128, feature_maps_128], dim=1)
        x_128 = self.convs['128_conv'](x_128)

        images_64 = self.convs['64_down'](x_128)
        feature_maps_64 = self.convs['64_fp'](feature_maps_64)
        x_64 = torch.cat([images_64, feature_maps_64], dim=1)
        x_64 = self.convs['64_conv'](x_64)

        x_32 = self.convs['32'](x_64)
        x_16 = self.convs['16'](x_32)
        img_code = self.convs['8'](x_16)

        pre_img_code = self.pre_encode(img_code)
        pair_disc_out = self.pair_disc(text_code, pre_img_code)
        local_img_disc_out = self.local_img_disc(img_code)

        return pair_disc_out, local_img_disc_out

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    generator = HDGANGenerator()
    generator.summary()

    discriminator = HDGANDiscriminator()
    discriminator.summary()








