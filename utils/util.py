import os


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