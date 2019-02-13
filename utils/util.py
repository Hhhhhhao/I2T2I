import os
import torch


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def word_list(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding words as a list.
    """
    word_list = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            word_list.append(word)
    return word_list


def clean_sentence(word_idx_list, vocab):
    """Take a list of word ids and a vocabulary from a dataset as inputs
    and return the corresponding sentence (as a single Python string).
    """
    sentence = []
    for i in range(len(word_idx_list)):
        vocab_id = word_idx_list[i]
        word = vocab.idx2word[vocab_id]
        if word == vocab.end_word:
            break
        if word != vocab.start_word:
            sentence.append(word)
    sentence = " ".join(sentence)
    return sentence


def convert_back_to_text(word_idx_array, vocab):
    from itertools import takewhile
    blacklist = [vocab.word2idx[word] for word in [vocab.start_word]]
    predicate = lambda word_id: vocab.idx2word[word_id] != vocab.end_word
    sampled_caption = [vocab.idx2word[word_id] for word_id in takewhile(predicate, word_idx_array) if word_id not in blacklist]

    sentence = ' '.join(sampled_caption)
    return sentence


def get_end_symbol_index(caption_list):
    if 1 in caption_list:
        return caption_list.index(1) + 1
    else:
        return len(caption_list)


def get_caption_lengths(captions_list):
    caption_lengths = [get_end_symbol_index(caption) for caption in captions_list]
    caption_lengths.sort(reverse=True)
    batch_captions = torch.zeros(len(captions_list), max(caption_lengths)).long()
    for i, cap in enumerate(captions_list):
        end = caption_lengths[i]
        batch_captions[i, :end] = torch.tensor(cap[:end]).long()
    return batch_captions, caption_lengths