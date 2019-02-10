import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from base import BaseModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super(MnistModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class EncoderCNN(BaseModel):
    """
    Encoder
    """

    def __init__(self, encode_image_size=4, embed_size=256):
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
        for c in list(self.resnet.children())[7:]:
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
        # print("hiddens shape:{}", hiddens[0].shape)
        outputs = self.linear(hiddens[0])

        # print("outputs shape {}".format(outputs.shape))
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
                    inputs = self.embedding(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]

    def pre_compute(self, features, gen_captions, eval_t, states=None):

        best_sample_nums = 5
        inputs = features.unsqueeze(1)

        if torch.cuda.is_available():
            gen_samples = gen_captions.type(torch.cuda.LongTensor)
        else:
            gen_samples = gen_captions.type(torch.LongTensor)

        prev_inputs = gen_samples[:, :eval_t]

        for i in range(eval_t):
            hiddens, states = self.lstm(inputs, states)
            inputs = self.embed(prev_inputs[:, i])
            inputs = inputs.unsqueeze(1)

        outputs = self.linear(hiddens.squeeze(1))
        outputs = self.softmax(outputs)
        predicted_indices = outputs.multinomial(best_sample_nums)

        return predicted_indices, states

    def rollout(self, gen_samples, t, max_len, states=None):
        """
            sample caption from a specific time t

        :param gen_samples:
        :param t: scalar
        :param max_len: scaler
        :param states: cell states, tuple
        :return:
        """

        sampled_ids = []

        if torch.cuda.is_available():
            gen_samples = gen_samples.type(torch.cuda.LongTensor)
        else:
            gen_samples = gen_samples.type(torch.LongTensor)

        inputs = self.embedding(gen_samples[:, t]).unsqueeze(1)
        for i in range(t, max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]

            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)
            inputs = inputs.unsqueeze(1)

        sampled_ids = torch.cat(sampled_ids, 0)
        sampled_ids = sampled_ids.view(-1, max_len-t)
        return sampled_ids


class EncoderRNN(BaseModel):
    def __init__(self, word_embed_size, hidden_size, vocab_size, output_feature_size, num_layers=1):
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
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.output_feature_size = output_feature_size

        self.embedding = nn.Embedding(vocab_size, word_embed_size)  # embedding layer
        self.lstm = nn.LSTM(word_embed_size, hidden_size, num_layers, bias=True, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_feature_size)  # linear layer to find scores over vocabulary
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
        # embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)

        packed = pack_padded_sequence(embeddings, caption_lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        # print("hiddens shape {}".format(hiddens[0].shape))
        padded = pad_packed_sequence(hiddens, batch_first=True)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(captions.size(0)), last_padded_indices, :]
        # print("hidden_outputs shape:{}".format(hidden_outputs.shape))
        outputs = self.linear(hidden_outputs)
        return outputs


class ImageCaptionGeneratorModel(BaseModel):
    def __init__(self, image_encode_size, image_embed_size, word_embed_size, lstm_hidden_size, vocab_size, lstm_num_layers=1):
        super(ImageCaptionGeneratorModel, self).__init__()
        self.image_encode_size = image_encode_size
        self.image_embed_size = image_embed_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.vocab_size = vocab_size

        self.encoder = EncoderCNN(self.image_encode_size, self.image_embed_size)
        self.decoder = DecoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.lstm_num_layers)

    def forward(self, images, captions, caption_lengths):
        self.features = self.encoder(images)
        outputs = self.decoder(self.features, captions, caption_lengths)
        return outputs

    def pre_compute(self, gen_captions, t):
        """
        pre compute the most likely vocabs idx and their states

        :param gen_captions: generated captions from decoder RNN (batch_size, max_len)
        :param t: time step t
        :return:
        """

        if self.features is None:
            raise RuntimeError('must do forward before calling this function')

        predicted_ids, saved_states = self.decoder.pre_compute(self.features, gen_captions, t)
        return predicted_ids, saved_states

    def rollout(self, gen_captions, t, saved_states):
        """
        rollout the remaining sentence part

        :param gen_captions: gen_captions: generated captions from decoder RNN (batch_size, max_len)
        :param t: time step t
        :param saved_states:
        :return:
        """

        if self.features is None:
            raise RuntimeError('must do forward before calling this function')

        max_len = gen_captions.size(1)
        sample_ids = self.decoder.rollout(gen_captions, t, max_len, states=saved_states)
        return sample_ids

    def sample(self, images):
        features = self.encoder(images)
        gen_captions_list = self.decoder.sample_beam_search(features)
        gen_captions = gen_captions_list[0]
        return gen_captions


class ImageCaptionDiscriminatorModel(BaseModel):
    def __init__(self,
                 image_encode_size,
                 word_embed_size,
                 lstm_hidden_size,
                 vocab_size,
                 image_feature_size,
                 sentence_feature_size,
                 lstm_num_layers=1):
        super(ImageCaptionDiscriminatorModel, self).__init__()
        self.image_encode_size = image_encode_size
        self.image_feature_size = image_feature_size
        self.sentence_feature_size = sentence_feature_size
        self.word_embed_size = word_embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.vocab_size = vocab_size
        self.lstm_num_layers = lstm_num_layers

        self.image_encoder = EncoderCNN(self.image_encode_size, self.image_feature_size)
        self.sentence_encoder = EncoderRNN(self.word_embed_size, self.lstm_hidden_size, self.vocab_size, self.sentence_feature_size, self.lstm_num_layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images, captions, caption_lengths):
        """ Calculate reward score: r = logistic(dot_prod(f, h))"""

        image_features = self.image_encoder(images)
        sentence_hidden_outputs = self.sentence_encoder(captions, caption_lengths)
        dot_product = torch.bmm(image_features.unsqueeze(1), sentence_hidden_outputs.unsqueeze(1).transpose(2,1)).squeeze()

        return self.sigmoid(dot_product)


if __name__ == '__main__':
    from data_loader import COCOCaptionDataLoader

    image_size = 128
    batch_size = 16

    data_loader = COCOCaptionDataLoader(
        data_dir='/Users/leon/Projects/I2T2I/data/coco/',
        which_set='val',
        image_size=image_size,
        batch_size=batch_size,
        num_workers=0,
        validation_split=0)

    for i, (image_ids, images, captions, caption_lengths) in enumerate(data_loader):
        print("done")
        break

    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)


    generator = ImageCaptionGeneratorModel(
        image_encode_size=4,
        word_embed_size=256,
        image_embed_size=256,
        lstm_hidden_size=512,
        vocab_size=len(data_loader.dataset.vocab)
    )

    discriminator = ImageCaptionDiscriminatorModel(
        image_encode_size=4,
        word_embed_size=256,
        lstm_hidden_size=512,
        image_feature_size=256,
        sentence_feature_size=256,
        vocab_size=len(data_loader.dataset.vocab))

    outputs = generator(images, captions, caption_lengths)
    score = discriminator(images, captions, caption_lengths)
    print('type(features):', type(outputs))
    print('features.shape:', outputs.shape)
    print('type(features):', type(score))
    print('features.shape:', score.shape)





