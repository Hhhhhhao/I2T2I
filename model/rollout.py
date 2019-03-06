# -*- coding:utf-8 -*-

import copy

import torch
import torch.nn.functional as F
from utils import get_caption_lengths
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Rollout:
    """Roll-out policy"""

    def __init__(self, max_sentence_length):
        self.lstm = None
        self.embedding = None
        self.max_sentence_length = max_sentence_length
        self.output_linear = None

    def reward(self, images, generated_captions, states, monte_carlo_count, evaluator, steps=1):
        assert monte_carlo_count % steps == 0, "Monte Carlo Count can't be divided by Steps"
        monte_carlo_count //= steps

        with torch.no_grad():
            batch_size = images.size(0)
            if torch.cuda.is_available():
                result = torch.zeros(batch_size, 1).cuda()
            else:
                result = torch.zeros(batch_size, 1)

            remaining = self.max_sentence_length - generated_captions.shape[1]
            h, c = states
            generated_captions = generated_captions.repeat(monte_carlo_count, 1)
            for _ in range(steps):
                states = (h.repeat(1, monte_carlo_count, 1), c.repeat(1, monte_carlo_count, 1))
                inputs = generated_captions[:, -1].unsqueeze(1)
                current_captions = generated_captions

                if torch.cuda.is_available():
                    inputs = inputs.cuda()

                inputs = self.embedding(inputs)

                for i in range(remaining):
                    hidden, states = self.lstm(inputs, states)
                    outputs = self.output_linear(hidden.squeeze(1))
                    outputs = F.softmax(outputs, -1)
                    predicted = outputs.multinomial(1)
                    predicted = predicted.long()
                    # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    current_captions = torch.cat([current_captions, predicted], dim=1)
                    inputs = self.embedding(predicted)
                caption_list = current_captions.data.clone()
                caption_list = caption_list.tolist()
                captions, caption_lengths = get_caption_lengths(caption_list)
                captions.to(device)
                # caption_lengths = [self.max_sentence_length] * current_captions.size(0)
                # captions = current_captions
                reward = evaluator.forward(images, captions, caption_lengths)
                reward = reward.view(batch_size, monte_carlo_count, -1).sum(1)
                result += reward
                result /= monte_carlo_count
            return result

    def update(self, original_model):
        self.embedding = copy.deepcopy(original_model.decoder.embedding)
        self.lstm = copy.deepcopy(original_model.decoder.lstm)
        self.lstm.flatten_parameters()
        self.output_linear = copy.deepcopy(original_model.decoder.linear)


