import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from utils.util import word_list
from eval_metrics.tokenizer.ptbtokenizer import PTBTokenizer



smoothing = SmoothingFunction()


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)





