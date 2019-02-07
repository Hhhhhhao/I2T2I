import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchgan
from torch.nn.utils.rnn import pack_padded_sequence
from base import BaseModel
from torch.autograd import Variable
from torchgan.models import Generator, Discriminator, DCGANGenerator, DCGANDiscriminator
from torchgan.layers import SpectralNorm2d, ResidualBlockTranspose2d
from math import ceil, log2
from collections import OrderedDict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CAEmbedding(BaseModel):
    def __init__(self, text_dim, embed_dim):
        super(CAEmbedding, self).__init__()

        self.text_dim = text_dim
        self.embed_dim = embed_dim
        self.linear = nn.Linear(self.text_dim, self.embed_dim*2, bias=True)
        self.relu = nn.ReLU()

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


class HDGANGenerator(DCGANGenerator):
    def __init__(self,
                 text_embed_dim=1024,
                 ca_code_dim=128,
                 noise_dim=128,
                 image_size=256,
                 image_channels=3,
                 discriminator_at=['64', '128', '256'],
                 step_channels=64,
                 batchnorm=True,
                 nonlinearity=None,
                 last_nonlinearity=None,
                 label_type='none'):
        super(HDGANGenerator, self).__init__(
                encoding_dims=ca_code_dim+noise_dim,
                out_size=image_size,
                out_channels=image_channels,
                step_channels=step_channels,
                batchnorm=batchnorm,
                nonlinearity=nonlinearity,
                last_nonlinearity=last_nonlinearity,
                label_type=label_type)
        self.intermediate_output = discriminator_at
        self.ca_embedding = CAEmbedding(text_embed_dim, ca_code_dim)

    def forward(self, text_embedding, noise):
        """Calculates the output tensor on passing the encoding ``x`` through the Generator.

        Args:
            noise (torch.Tensor): A 2D torch tensor of the encoding sampled from a probability
                distribution.
            feature_matching (bool, optional): Returns the activation from a predefined intermediate
                layer.

        Returns:
            A 4D torch.Tensor of the generated image.
        """

        output = OrderedDict()

        c_code, mean, log_var = self.ca_embedding(text_embedding)
        # print("c shape:{}".format(c_code.shape))
        x = torch.cat((noise, c_code), 1)
        # print("x shape:{}".format(x.shape))
        x = x.view(-1, x.size(1), 1, 1)
        # print("input shape:{}".format(x.shape))
        for i in range(len(self.model)):
            x = self.model[i](x)
            if str(x.shape[-1]) in self.intermediate_output:
                output[str(x.shape[-1])] = x
            # print("middle {} shape:{}".format(i, x.shape))
            # print(str(x.shape[-1]))
        return output, mean, log_var


class HDGANDiscriminator(DCGANDiscriminator):
    def __init__(self,
                 text_embed_dim=1024,
                 reduced_text_embed_dim=128,
                 image_size=256,
                 image_channels=3,
                 step_channels=64,
                 batchnorm=True,
                 nonlinearity=None,
                 last_nonlinearity=None,
                 label_type='none'
                 ):
        super(HDGANDiscriminator, self).__init__(
            in_size=image_size,
            in_channels=image_channels,
            step_channels=step_channels,
            batchnorm=batchnorm,
            nonlinearity=nonlinearity,
            last_nonlinearity=last_nonlinearity,
            label_type=label_type)
        self.image_size = image_size
        self.linear = nn.Linear(text_embed_dim, reduced_text_embed_dim, bias=True)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, images, text_embeddings):
        image_size = images.size()[-1]
        assert image_size == self.image_size, "wrong input size {} in discriminator".format(image_size)

        disc_output = self.model[:-1](images)
        print(disc_output.shape)

#
#
# class HDGANDiscriminator(BaseModel):
#     def __init__(self,
#                  text_embed_dim=1024,
#                  reduced_text_embed_dim=128,
#                  image_size=256,
#                  image_channels=3,
#                  step_channels=64,
#                  discriminator_at=['64', '128', '256'],
#                  batchnorm=True,
#                  nonlinearity=None,
#                  last_nonlinearity=nn.Sigmoid(),
#                  label_type='none'
#                  ):
#         super(HDGANDiscriminator).__init__()
#
#         self.discriminators = OrderedDict()
#         self.discriminator_at = discriminator_at
#         self.linear = nn.Linear(text_embed_dim, reduced_text_embed_dim, bias=True)
#         self.activation = nn.LeakyReLU(0.2, True)
#
#         for i, in_size in enumerate(discriminator_at):
#             if i == 2 and in_size == str(image_size):
#                 in_channels=image_channels
#             else:
#                 in_channels=(i-1)*step_channels
#
#             self.discriminators[in_size] = DCGANDiscriminator(
#                 in_size=in_size,
#                 in_channels=in_channels,
#                 step_channels=step_channels,
#                 batchnorm=batchnorm,
#                 nonlinearity=nonlinearity,
#                 last_nonlinearity=last_nonlinearity,
#                 label_type=label_type
#             )
#
#             print("discriminator:{}".format(in_size))
#             print(self.discriminators[in_size])
#
#
#     def forward(self, images, text_embedding):
#
#         image_size = images.size()[-1]
#         assert image_size in self.discriminator_at, "wrong input size {} in discriminator".format(image_size)
#
#         disc_output = self.discriminators[str(image_size)][:-1](images)
#         print(disc_output)













class EncoderCNN(BaseModel):
    """
    Encoder
    """

    def __init__(self, image_size=256, encode_image_size=4, embed_size=256):
        super(EncoderCNN, self).__init__()

        resnet = torchvision.models.resnet34(pretrained=True)

        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((encode_image_size, encode_image_size))
        # Resize image to fixed size to allow input images of variable size
        self.linear = nn.Linear(encode_image_size**2 * 512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        self.fine_tune()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.01)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """

        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)  # (batch_size, embed_size)

        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune:
        """

        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class DecoderRNN(BaseModel):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        """
        Set the hyper-parameters and build the layers.

        :param embed_size: word embedding size
        :param hidden_size: hidden unit size of LSTM
        :param vocab_size: size of vocabulary (output of the network)
        :param num_layers:
        :param dropout: use of drop out
        """
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size) # embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)   # linear layer to find scores over vocabulary
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions, caption_lengths):
        """
        Decode image feature vectors and generate captions.

        :param features: encoded images, a tensor of dimension (batch_size, encoded_image_size, encoded_image_size, 2048)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores of vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states) # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, features, states=None, max_len=20, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted
        sentences. This is the beam search approach.
        """
        # Top word idx sequences and their corresponding inputs and states
        inputs = features.unsqueeze(1)
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]


class ImageCaptionModel(BaseModel):
    def __init__(self, image_size, image_encode_size, image_embed_size, word_embed_size, lstm_hidden_size, vocab_size, lstm_num_layers=1):
        super(ImageCaptionModel, self).__init__()
        self.image_size = image_size
        self.image_encode_size = image_encode_size
        self.image_embed_size = image_embed_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size

        self.encoder = EncoderCNN(self.image_size, self.image_encode_size, self.image_embed_size)
        self.decoder = DecoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.lstm_num_layers)

    def forward(self, images, captions, caption_lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, caption_lengths)
        return outputs

    def sample(self, images, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
        sampled_ids = []
        features = self.encoder(images)
        inputs = features.unsqueeze(1)
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states) # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, images, states=None, max_len=20, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted
        sentences. This is the beam search approach.
        """
        features = self.encoder(images)
        # Top word idx sequences and their corresponding inputs and states
        inputs = features.unsqueeze(1)
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]


if __name__ == '__main__':

    from data_loader import TextEmbeddingDataLoader
    import numpy as np

    birds_data_loader = TextEmbeddingDataLoader(
        data_dir='/Users/leon/Projects/I2T2I/data/',
        dataset_name="flowers",
        which_set='train',
        image_size=256,
        batch_size=16,
        num_workers=0
    )

    generator = HDGANGenerator(
                 text_embed_dim=1024,
                 ca_code_dim=128,
                 noise_dim=128)
    print(generator.model)

    discriminator = HDGANDiscriminator(
                 text_embed_dim=1024,
                 reduced_text_embed_dim=128,
                 image_size=256,
                 image_channels=3)


    for i, sample in enumerate(birds_data_loader):
        images = sample["right_image"]
        text_embeddings = sample["right_embed"]
        noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (images.shape[0], 128))))
        generator(text_embeddings, noise)

        discriminator(images, text_embeddings)








