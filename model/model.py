import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.distributions import Normal
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base import BaseModel
from model.rollout import Rollout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderCNN(BaseModel):
    """
    Encoder
    """

    def __init__(self, image_embed_size=256):
        super(EncoderCNN, self).__init__()

        adaptive_pool_size = 12
        resnet = torchvision.models.resnet34(pretrained=True)

        # Remove average pooling layers
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((adaptive_pool_size, adaptive_pool_size))
        self.fc_in_features = 256 * adaptive_pool_size ** 2
        self.linear = nn.Linear(self.fc_in_features, image_embed_size)
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

        return features

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.
        :param fine_tune:
        """

        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[6:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class EncoderRNN(BaseModel):
    def __init__(self, word_embed_size, sentence_embed_size, lstm_hidden_size, vocab_size, num_layers=1):
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
        self.lstm = nn.LSTM(word_embed_size, lstm_hidden_size, num_layers, bias=True, batch_first=True)
        self.linear = nn.Linear(lstm_hidden_size, sentence_embed_size)  # linear layer to find scores over vocabulary
        # self.activation = nn.LeakyReLU(0.2)
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
        self.lstm.flatten_parameters()
        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        caption_lengths = caption_lengths.to("cpu").tolist()
        total_length = captions.size(1)
        packed = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        # print("hiddens shape {}".format(hiddens[0].shape))
        padded = pad_packed_sequence(hiddens, batch_first=True, total_length=total_length)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(captions.size(0)), last_padded_indices, :]
        # print("hidden_outputs shape:{}".format(hidden_outputs.shape))
        outputs = self.linear(hidden_outputs)
        # outputs = self.activation(outputs)
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
        self.lstm.flatten_parameters()
        # Embedding
        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        caption_lengths = caption_lengths.to("cpu").tolist()

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

    def sample_beam_search(self, features, max_len=20, beam_width=3, states=None):
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
        self.features_linear = nn.Linear(self.image_embed_size + noise_dim, self.image_embed_size)
        self.decoder = DecoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.lstm_num_layers)
        self.rollout = Rollout(max_sentence_length)

        self.activation = nn.LeakyReLU(0.2)

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

    def caption_forward(self, images, captions, caption_lengths):
        image_features = self.encoder(images)
        features = self.activation(image_features)
        outputs = self.decoder(features, captions, caption_lengths)
        return image_features, outputs

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
        self.decoder.lstm.flatten_parameters()
        batch_size = images.size(0)
        image_features = self.encoder(images)

        features = self.get_feature_linear_output(image_features)
        # initialize hiddens states of lstm
        states = self.init_states_from_features(features.unsqueeze(1))
        # initialize inputs of start symbol
        inputs = torch.zeros((batch_size, 1)).long()
        current_generated_captions = torch.LongTensor(inputs)
        rewards = torch.zeros(batch_size, self.max_sentence_length)
        props = torch.zeros(batch_size, self.max_sentence_length)

        if torch.cuda.is_available():
            inputs = inputs.cuda()
            rewards = rewards.cuda()
            props = props.cuda()
            current_generated_captions.cuda()

        inputs = self.decoder.embedding(inputs)

        if torch.cuda.is_available():
            inputs = inputs.cuda()

        self.rollout.update(self)

        for i in range(self.max_sentence_length):

            hiddens, states = self.decoder.lstm(inputs, states)
            # squeeze the hidden output size from (batch_siz, 1, hidden_size) to (batch_size, hidden_size)
            outputs = self.decoder.linear(hiddens.squeeze(1))

            # outputs of size (batch_size, vocab_size)
            outputs = F.softmax(outputs, -1)

            # use multinomial to random sample
            # predicted = outputs.argmax(1)
            # predicted = (predicted.unsqueeze(1)).long()
            predicted = outputs.multinomial(1)

            # if torch.cuda.is_available():
            #   predicted = predicted.cuda()
            prop = torch.gather(outputs, 1, predicted)
            # prop is a 1D tensor
            props[:, i] = prop.view(-1)

            # embed the next inputs, unsqueeze is required cause of shape (batch_size, vocab_size)
            current_generated_captions = torch.cat([current_generated_captions, predicted.cpu()], dim=1)
            inputs = self.decoder.embedding(predicted)

            reward = self.rollout.reward(images, current_generated_captions, states, monte_carlo_count, evaluator)
            rewards[:, i] = reward.view(-1)
        return rewards, props

    def sample(self, features, states=None):
        return self.decoder.sample(features, states, self.max_sentence_length)

    def sample_beam_search(self, features, beam_width=3, states=None):
        return self.decoder.sample_beam_search(features, self.max_sentence_length, beam_width, states)


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


    generator = ConditionalGenerator(
        word_embed_size=512,
        image_embed_size=512,
        lstm_hidden_size=1024,
        vocab_size=len(data_loader.dataset.vocab)
    )

    discriminator = Evaluator(
        word_embed_size=512,
        image_embed_size=512,
        sentence_embed_size=512,
        lstm_hidden_size=1024,
        vocab_size=len(data_loader.dataset.vocab))

    image_features, outputs = generator(images, captions, caption_lengths)
    score = discriminator(images, captions, caption_lengths)
    # rewards, props = generator.reward_forward(image_features, discriminator)
    rewards, props = generator.reward_forward(images, discriminator)
    print('type(features):', type(outputs))
    print('features.shape:', outputs.shape)
    print('type(features):', type(score))
    print('features.shape:', score.shape)





