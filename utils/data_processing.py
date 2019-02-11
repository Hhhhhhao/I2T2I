import nltk
import os
import pickle
import string
import h5py
import sys
import string
import numpy as np
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
sys.path.append(os.path.join(dirname, '/data/coco/cocoapi/PythonAPI'))
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import Counter
from autocorrect import spell
from nltk.corpus import wordnet as WN
from nltk.corpus import stopwords
stop_words_en = set(stopwords.words('english'))


class COCOVocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file=os.path.join(dirname, "/data/coco/vocab.pkl"),
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file=os.path.join(dirname, "/data/coco/annotations/captions_train2017.json"),
        vocab_from_file=False):

        """
        Initialize the vocabulary.
            Paramters:
              vocab_threshold: Minimum word count threshold.
              vocab_file: File containing the vocabulary.
              start_word: Special word denoting sentence start.
              end_word: Special word denoting sentence end.
              unk_word: Special word denoting unknown words.
              annotations_file: Path for train annotation file.
              vocab_from_file: If False, create vocab from scratch & override any
                               existing vocab_file. If True, load vocab from from
                               existing vocab_file, if it exists.
        """
        self.vocab_threshold = vocab_threshold
        self.vocab_file = vocab_file
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word
        self.annotations_file = annotations_file
        self.vocab_from_file = vocab_from_file
        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file or build it from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers
        (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers
        (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary
        that meet or exceed the threshold."""
        coco = COCO(self.annotations_file)
        counter = Counter()
        ids = coco.anns.keys()
        for i, id in enumerate(ids):
            caption = str(coco.anns[id]["caption"])
            caption = text_clean(caption)
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            counter.update(tokens)

            if i % 100000 == 0:
                print("[%d/%d] Tokenizing captions..." % (i, len(ids)))

        words = [word for word, cnt in counter.items()
                 if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        dataset_name,
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        vocab_from_file=False,
        data_dir=dirname + '/data/',
                 ):

        """
        Initialize the vocabulary.
            Paramters:
              vocab_threshold: Minimum word count threshold.
              vocab_file: File containing the vocabulary.
              start_word: Special word denoting sentence start.
              end_word: Special word denoting sentence end.
              unk_word: Special word denoting unknown words.
              dataset_name: The name of dataset to be used.
              data_dir: The main directory of datasets
        """
        self.vocab_threshold = vocab_threshold
        self.start_word = start_word
        self.end_word = end_word
        self.unk_word = unk_word

        if vocab_from_file:
            raise ValueError("cannot read h5 data from pickle, please try generate dictionary")

        self.vocab_from_file = vocab_from_file

        assert dataset_name in {'birds', 'flowers'}, "Wrong dataset name: {}".format(dataset_name)
        self.dataset_name = dataset_name

        if os.path.exists(data_dir):
            self.vocab_file = os.path.join(data_dir, '{}/vocab.pkl'.format(dataset_name))
            self.h5_file = os.path.join(data_dir, '{}/{}.hdf5'.format(dataset_name, dataset_name))
        else:
            raise ValueError("data directory do not exist")

        self.get_vocab()

    def get_vocab(self):
        """Load the vocabulary from file or build it from scratch."""
        if os.path.exists(self.vocab_file) & self.vocab_from_file:
            with open(self.vocab_file, "rb") as f:
                vocab = pickle.load(f)
                self.word2idx = vocab.word2idx
                self.idx2word = vocab.idx2word
            print("Vocabulary successfully loaded from vocab.pkl file!")
        else:
            self.build_vocab()
            with open(self.vocab_file, "wb") as f:
                pickle.dump(self, f)

    def build_vocab(self):
        """Populate the dictionaries for converting tokens to integers
        (and vice-versa)."""
        self.init_vocab()
        self.add_word(self.start_word)
        self.add_word(self.end_word)
        self.add_word(self.unk_word)
        self.add_captions()

    def init_vocab(self):
        """Initialize the dictionaries for converting tokens to integers
        (and vice-versa)."""
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        """Add a token to the vocabulary."""
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_captions(self):
        """Loop over training captions and add all tokens to the vocabulary
        that meet or exceed the threshold."""
        if os.path.exists(self.h5_file):
            self.data = h5py.File(self.h5_file, mode='r')
            ids = [str(k) for k in self.data['train'].keys()]
        else:
            raise ValueError("data h5 file do not exist")

        counter = Counter()
        for i, id in enumerate(tqdm(ids)):
            caption = str(np.array(self.data['train'][id]['txt']))
            caption = text_clean(caption)
            tokens = nltk.tokenize.word_tokenize(caption.lower())
            # tokens = [word for word in tokens]
            counter.update(tokens)

        words = [word for word, cnt in counter.items()
                 if cnt >= self.vocab_threshold]

        for i, word in enumerate(words):
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.unk_word]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def SpellChecker(token):
    strip = token.rstrip()
    if not WN.synsets(strip) and not (strip in string.punctuation):
        if strip in stop_words_en:
            return token
        else:
            print("wrong word:{}".format(token))
            abc = spell(token)
            print("right word:{}".format(abc))
            return abc
    else:
        return token


def remove_punctuation(text_original):
    translator = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text_original.translate(translator)
    return(text_no_punctuation)


def text_clean(text_original):
    text = remove_punctuation(text_original)
    return text


if __name__ == "__main__":
    # import string

    # print(list(string.punctuation))

    vocab = Vocabulary(vocab_threshold=4,
                       dataset_name='birds')

    word2idx = vocab.word2idx
    print(word2idx)
    print(word2idx.keys())
    idx2word = vocab.idx2word
    print(idx2word)