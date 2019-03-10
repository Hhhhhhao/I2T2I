import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions import Normal
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models
import torch.utils.model_zoo as model_zoo
from base import BaseModel
from model.rollout import Rollout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv1x1(in_planes, out_planes, bias=False):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=bias)


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


class DAMSM_CNN_Encoder(BaseModel):
    def __init__(self,
                 embedding_size=256):
        super(DAMSM_CNN_Encoder, self).__init__()
        self.embedding_size = embedding_size  # define a uniform ranker

        model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        model.load_state_dict(model_zoo.load_url(url))
        for param in model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        self.define_module(model)
        self.init_trainable_weights()

    def define_module(self, model):
        self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
        self.Mixed_5b = model.Mixed_5b
        self.Mixed_5c = model.Mixed_5c
        self.Mixed_5d = model.Mixed_5d
        self.Mixed_6a = model.Mixed_6a
        self.Mixed_6b = model.Mixed_6b
        self.Mixed_6c = model.Mixed_6c
        self.Mixed_6d = model.Mixed_6d
        self.Mixed_6e = model.Mixed_6e
        self.Mixed_7a = model.Mixed_7a
        self.Mixed_7b = model.Mixed_7b
        self.Mixed_7c = model.Mixed_7c

        self.emb_features = conv1x1(768, self.embedding_size)
        self.emb_cnn_code = nn.Linear(2048, self.embedding_size)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        features = None
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False)(x)
        # 299 x 299 x 3
        x = self.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.Conv2d_4a_3x3(x)
        # 71 x 71 x 192

        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)
        # 35 x 35 x 288

        x = self.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)
        # 17 x 17 x 768

        # image region features
        features = x
        # 17 x 17 x 768

        x = self.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)
        # 8 x 8 x 2048
        x = F.avg_pool2d(x, kernel_size=8)
        # 1 x 1 x 2048
        # x = F.dropout(x, training=self.training)
        # 1 x 1 x 2048
        x = x.view(x.size(0), -1)
        # 2048

        # global image features
        cnn_code = self.emb_cnn_code(x)
        # 512
        if features is not None:
            features = self.emb_features(features)

        # features of size: batch_size * embedding_size * 17 * 17
        # cnn_code of size: batch_size * embedding_size
        return features, cnn_code


class DAMSM_RNN_Encoder(BaseModel):
    def __init__(self,
                 vocab_size,
                 word_embed_size=256,
                 lstm_hidden_size=256,
                 lstm_num_layers=1,
                 drop_prob=0.5,
                 bidirectional=True):

        super(DAMSM_RNN_Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.drop_prob = drop_prob
        self.bidirectional = bidirectional

        if bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        # number of features in the hidden state
        self.lstm_hidden_size = lstm_hidden_size // self.num_directions

        self.define_module()
        self.init_weights()

    def define_module(self):
        self.embedding = nn.Embedding(self.vocab_size, self.word_embed_size)  # embedding layer
        self.drop = nn.Dropout(self.drop_prob)
        # dropout: If non-zero, introduces a dropout layer on
        # the outputs of each RNN layer except the last layer
        self.lstm = nn.LSTM(self.word_embed_size,
                           self.lstm_hidden_size,
                           self.lstm_num_layers,
                           batch_first=True,
                           bidirectional=self.bidirectional)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        # Do not need to initialize RNN parameters, which have been initialized
        # http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
        # self.decoder.weight.data.uniform_(-initrange, initrange)
        # self.decoder.bias.data.fill_(0)

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if torch.cuda.is_available():
            return (Variable(weight.new(self.lstm_num_layers * self.num_directions,
                                        bsz, self.lstm_hidden_size).zero_()).cuda(),
                    Variable(weight.new(self.lstm_num_layers * self.num_directions,
                                        bsz, self.lstm_hidden_size).zero_()).cuda())
        else:
            return (Variable(weight.new(self.lstm_num_layers * self.num_directions,
                                        bsz, self.lstm_hidden_size).zero_()),
                    Variable(weight.new(self.lstm_num_layers * self.num_directions,
                                        bsz, self.lstm_hidden_size).zero_()))

    def forward(self, captions, caption_lengths, states=None):
        # input: torch.LongTensor of size batch x n_steps
        # --> emb: batch x n_steps x ninput
        if torch.cuda.is_available():
            self.lstm.flatten_parameters()

        total_length = captions.size(1)
        emb = self.drop(self.embedding(captions))
        #
        caption_lengths = caption_lengths.to('cpu').tolist()

        emb = pack_padded_sequence(emb, caption_lengths, batch_first=True)
        # #hidden and memory (num_layers * num_directions, batch, hidden_size):
        # tensor containing the initial hidden state for each element in batch.
        # #output (batch, seq_len, hidden_size * num_directions)
        # #or a PackedSequence object:
        # tensor containing output features (h_t) from the last layer of RNN
        output, hidden = self.lstm(emb, states)
        # PackedSequence object
        # --> (batch, seq_len, hidden_size * num_directions)
        output = pad_packed_sequence(output, batch_first=True, total_length=total_length)[0]
        # output = self.drop(output),
        # --> batch x hidden_size*num_directions x seq_len
        words_emb = output.transpose(1, 2)
        # --> batch x num_directions*hidden_size
        sent_emb = hidden[0].transpose(0, 1).contiguous()
        sent_emb = sent_emb.view(-1, self.lstm_hidden_size * self.num_directions)
        return words_emb, sent_emb


class DAMSM(BaseModel):
    def __init__(self,
                 vocab_size,
                 word_embed_size=256,
                 embedding_size=1024):
        super(DAMSM, self).__init__()

        self.cnn_encoder = DAMSM_CNN_Encoder(embedding_size)
        self.rnn_encoder = DAMSM_RNN_Encoder(vocab_size, word_embed_size, embedding_size)

    def forward(self, images, captions, caption_lengths):
        # words_features: batch_size x embedding_size x 17 x 17
        # sent_code: batch_size x embedding_size
        image_features, image_emb = self.cnn_encoder(images)
        batch_size = images.size(0)

        states = self.rnn_encoder.init_hidden(batch_size)
        words_emb, sent_emb = self.rnn_encoder(captions, caption_lengths, states)

        return image_features, image_emb, words_emb, sent_emb


class EncoderCNN(BaseModel):
    """
    Encoder
    """

    def __init__(self, image_embed_size=512):
        super(EncoderCNN, self).__init__()

        adaptive_pool_size = 8
        resnet = torchvision.models.resnet34(pretrained=True)

        # Remove linear and pool layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))
        self.fc_in_features = 512 * adaptive_pool_size ** 2
        # Resize image to fixed size to allow input images of variable size
        self.linear = nn.Linear(self.fc_in_features, image_embed_size)
        # self.activation = nn.LeakyReLU(0.2)
        # self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
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
        # features = self.activation(features)
        # features = self.bn(features)  # (batch_size, embed_size)

        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune:
        """

        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[7:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class EncoderRNN(BaseModel):
    def __init__(self,
                 word_embed_size,
                 sentence_embed_size,
                 lstm_hidden_size,
                 vocab_size,
                 num_layers=1):
        """
        Set the hyper-parameters and build the layers.
        :param embed_size: word embedding size
        :param hidden_size: hidden unit size of LSTM
        :param vocab_size: size of vocabulary (output of the network)
        :param num_layers:
        :param dropout: use of drop out
        """
        super(EncoderRNN, self).__init__()
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.sentence_embed_size = sentence_embed_size

        self.embedding = nn.Embedding(vocab_size, word_embed_size)  # embedding layer
        self.lstm = nn.LSTM(word_embed_size, lstm_hidden_size, num_layers, bias=True, batch_first=True, bidirectional=self.bidirectional)
        self.linear = nn.Linear(lstm_hidden_size, sentence_embed_size)  # linear layer to find scores over vocabulary
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, captions, caption_lengths):
        """
        Decode image feature vectors and generate captions.
        :param features: encoded images, a tensor of dimension (batch_size, encoded_image_size, encoded_image_size, 2048)
        :param captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores of vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)

        packed = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        padded = pad_packed_sequence(hiddens, batch_first=True)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(captions.size(0)), last_padded_indices, :]
        outputs = self.linear(hidden_outputs)
        return outputs


class DecoderRNN(BaseModel):
    def __init__(self, word_embed_size, lstm_hidden_size, vocab_size, num_layers=1):
        """
        Set the hyper-parameters and build the layers.
        :param embed_size: word embedding size
        :param hidden_size: hidden unit size of LSTM
        :param vocab_size: size of vocabulary (output of the network)
        :param num_layers:
        :param dropout: use of drop out
        """
        super(DecoderRNN, self).__init__()
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, word_embed_size) # embedding layer
        self.lstm = nn.LSTM(word_embed_size, lstm_hidden_size, num_layers, bias=True, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, vocab_size)   # linear layer to find scores over vocabulary
        self.init_weights()

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.linear.weight.data.uniform_(-0.1, 0.1)

    def forward(self, features, captions, caption_lengths):
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
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, features, max_len=20, beam_width=5, states=None):
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
                    inputs = self.embedding(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]


class ImageCaptionModel(BaseModel):
    def __init__(self, image_embed_size, word_embed_size, lstm_hidden_size, vocab_size, lstm_num_layers=1):
        super(ImageCaptionModel, self).__init__()
        self.image_embed_size = image_embed_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size

        self.encoder = EncoderCNN(self.image_embed_size)
        self.decoder = DecoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.lstm_num_layers)

    def forward(self, images, captions, caption_lengths):
        features = self.encoder(images)
        outputs = self.decoder(features, captions, caption_lengths)
        return outputs

    def sample(self, features, max_len=20, states=None):
        return self.decoder.sample(features, max_len, states)

    def sample_beam_search(self, features, max_len=20, beam_width=5, states=None):
        return self.decoder.sample_beam_search(features, max_len, beam_width, states)


class ConditionalGenerator(BaseModel):

    def __init__(self,
                 image_embed_size=512,
                 word_embed_size=512,
                 lstm_hidden_size=1024,
                 noise_dim=128,
                 vocab_size=10000,
                 lstm_num_layers=1,
                 max_sentence_length=20):
        super(ConditionalGenerator, self).__init__()
        self.image_embed_size =image_embed_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size
        self.max_sentence_length = max_sentence_length
        # noise variable
        self.distribution = Normal(Variable(torch.zeros(noise_dim)), Variable(torch.ones(noise_dim)))

        # image feature encoder
        self.encoder = EncoderCNN(self.image_embed_size)
        self.features_linear = nn.Sequential(
            nn.Linear(self.image_embed_size + noise_dim, self.image_embed_size),
            nn.LeakyReLU(0.2)
        )
        self.decoder = DecoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.lstm_num_layers)
        self.rollout = Rollout(max_sentence_length)

    def get_feature_linear_output(self, image_features):
        rand = self.distribution.sample((image_features.shape[0],))

        if torch.cuda.is_available():
            rand = rand.cuda()

        inputs = torch.cat((image_features, rand), 1)
        features = self.features_linear(inputs)

        if torch.cuda.is_available():
            return features.cuda()
        else:
            return features

    def init_states_from_features(self, features):
        _, states = self.decoder.lstm(features)

        if torch.cuda.is_available():
            return (states[0].cuda(), states[1].cuda())
        else:
            return states

    def forward(self, images, captions, caption_lengths):
        image_features = self.encoder(images)
        features = self.get_feature_linear_output(image_features)
        outputs = self.decoder(features, captions, caption_lengths)
        return image_features, outputs

    def reward_forward(self, images, evaluator, monte_carlo_count=18):
        '''

        :param image: image features from image encoder linear layer
        :param evaluator: evaluator model
        :param monte_carlo_count: monte carlo count
        :return:
        '''
        # self.decoder.lstm.flatten_parameters()
        batch_size = images.size(0)
        image_features = self.encoder(images)
        features = self.get_feature_linear_output(image_features)
        # initialize hiddens states of lstm
        states = self.init_states_from_features(features.unsqueeze(1))
        # initialize inputs of start symbol
        inputs = torch.zeros((batch_size, 1)).long()
        current_generated_captions = inputs
        rewards = torch.zeros(batch_size, self.max_sentence_length)
        props = torch.zeros(batch_size, self.max_sentence_length)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            rewards = rewards.cuda()
            props = props.cuda()
            # current_generated_captions.cuda()

        inputs = self.decoder.embedding(inputs)
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        self.rollout.update(self)

        for i in range(self.max_sentence_length):

            hiddens, states = self.decoder.lstm(inputs, states)
            # squeeze the hidden output size from (batch_siz, 1, hidden_size) to (batch_size, hidden_size)
            outputs = self.decoder.linear(hiddens.squeeze(1))
            outputs = F.softmax(outputs, -1)

            # use multinomial to random sample
            predicted = outputs.argmax(1)
            predicted = (predicted.unsqueeze(1)).long()

            # if torch.cuda.is_available():
            #   predicted = predicted.cuda()

            prop = torch.gather(outputs, 1, predicted)
            # prop is a 1D tensor
            props[:, i] = prop.view(-1)
            # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
            current_generated_captions = torch.cat([current_generated_captions, predicted.cpu()], dim=1)
            inputs = self.decoder.embedding(predicted)
            reward = self.rollout.reward(images, current_generated_captions, states, monte_carlo_count, evaluator)
            rewards[:, i] = reward.view(-1)
        return rewards, props

    def sample(self, features, states=None):
        return self.decoder.sample(features, states, self.max_sentence_length)

    def sample_beam_search(self, features, beam_width=5, states=None):
        return self.decoder.sample_beam_search(features, self.max_sentence_length, beam_width, states)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


class Evaluator(BaseModel):
    def __init__(self,
                 word_embed_size=512,
                 image_embed_size=512,
                 sentence_embed_size=512,
                 lstm_hidden_size=1024,
                 vocab_size=100000,
                 lstm_num_layers=1):
        super(Evaluator, self).__init__()
        self.image_embed_size = image_embed_size
        self.sentence_embed_size = sentence_embed_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.lstm_num_layers = lstm_num_layers

        # image encoder
        self.image_encoder = EncoderCNN(self.image_embed_size)
        self.sentence_encoder = EncoderRNN(self.word_embed_size,
                                           self.sentence_embed_size,
                                           self.lstm_hidden_size,
                                           self.vocab_size,
                                           self.lstm_num_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, captions, caption_lengths):
        """ Calculate reward score: r = logistic(dot_prod(f, h))"""
        image_features = self.image_encoder(images)

        if image_features.size(0) != captions.size(0):
            monte_carlo_count = int(captions.size(0) / image_features.size(0))
            image_features = image_features.repeat(monte_carlo_count, 1)

        sentence_features = self.sentence_encoder(captions, caption_lengths)
        dot_product = torch.bmm(image_features.unsqueeze(1), sentence_features.unsqueeze(1).transpose(2,1)).squeeze()
        return self.sigmoid(dot_product)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    from data_loader.data_loaders import TextImageDataLoader

    image_size = 128
    batch_size = 16

    data_loader = TextImageDataLoader(
        data_dir='/Users/leon/Projects/I2T2I/data/',
        dataset_name="birds",
        which_set='train',
        image_size=256,
        batch_size=16,
        num_workers=0)

    for i, data in enumerate(data_loader):
        print("done")
        images = data["right_images_128"]
        captions = data["right_captions"]
        caption_lengths = data["right_caption_lengths"]
        break

    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)
    print(caption_lengths)
    print([0] * 5)

    model = DAMSM(
        embedding_size=512,
        word_embed_size=256,
        vocab_size=len(data_loader.dataset.vocab),
    )

    # generator = ConditionalGenerator(
    #     word_embed_size=512,
    #     image_embed_size=512,
    #     lstm_hidden_size=1024,
    #     vocab_size=len(data_loader.dataset.vocab)
    # )
    #
    # discriminator = Evaluator(
    #     word_embed_size=512,
    #     image_embed_size=512,
    #     sentence_embed_size=512,
    #     lstm_hidden_size=1024,
    #     vocab_size=len(data_loader.dataset.vocab))

    # image_features, outputs = generator(images, captions, caption_lengths)
    # score = discriminator(images, captions, caption_lengths)
    # # rewards, props = generator.reward_forward(image_features, discriminator)
    # rewards, props = generator.reward_forward(images, discriminator)

    image_features, image_emb, words_emb, sent_emb = model(images, captions, caption_lengths)
    print('type(image_features):', type(image_features))
    print('image_features.shape:', image_features.shape)
    print('type(image_emb):', type(image_emb))
    print('image_emb.shape:', image_emb.shape)
    print('type(words_emb):', type(words_emb))
    print('words_emb.shape:', words_emb.shape)
    print('type(sent_emb):', type(sent_emb))
    print('sent_emb.shape:', sent_emb.shape)





