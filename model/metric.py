import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.util import word_list


smoothing = SmoothingFunction()

def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def bleu4(output, target, length, vocab):
    batch_bleu_4 = 0.0
    # Iterate over outputs. Note: outputs[i] is a caption in the batch
    # outputs[i, j, k] contains the model's predicted score i.e. how
    # likely the j-th token in the i-th caption in the batch is the
    # k-th token in the vocabulary.
    with torch.no_grad():
        caption_list = target.tolist()
        output_list = output.tolist()
        start_slice = 0
        end_slice = 0
        for i, len in enumerate(length):
            predicted_ids = []
            end_slice += len
            sentence_target = caption_list[start_slice:end_slice]
            sentence_output = output_list[start_slice:end_slice]
            start_slice = end_slice
            for scores in sentence_output:
                # Find the index of the token that has the max score
                predicted_ids.append(np.argmax(scores))
            # Convert word ids to actual words
            predicted_word_list = word_list(predicted_ids, vocab)
            caption_word_list = word_list(sentence_target, vocab)
            # Calculate Bleu-4 score
            batch_bleu_4 += sentence_bleu([caption_word_list],
                                          predicted_word_list,
                                          smoothing_function=smoothing.method1)
    return batch_bleu_4 / len(length)
