import nltk
import os
import pickle
import json
import h5py
import sys
import cv2
import numpy as np
sys.path.append('/Users/leon/Projects/I2T2I/data/coco/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from tqdm import tqdm
from collections import Counter


class Vocabulary(object):

    def __init__(self,
        vocab_threshold,
        vocab_file="/Users/leon/Projects/I2T2I/data/coco/vocab.pkl",
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        annotations_file="/Users/leon/Projects/I2T2I/data/coco/annotations/captions_train2017.json",
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


def create_coco_h5_files(image_dir, vocab_threshold, output_dir):

    if os.path.exists(image_dir):
        print('start create hdf5 files for COCO dataset')
    else:
        raise ValueError('cannot find the images directory')

    train_images_dir = os.path.join(image_dir, 'train')
    val_images_dir = os.path.join(image_dir, 'val')
    test_images_dir = os.path.join(image_dir, 'test')
    train_captions_file_path = os.path.join(image_dir, 'annotations/captions_train2017.json')
    val_captions_file_path = os.path.join(image_dir, 'annotations/captions_val2017.json')
    vocab = Vocabulary(vocab_threshold, vocab_from_file=True)

    base_filename = 'coco_' + str(vocab_threshold) + '_min_word_freq'

    for split in ['val', 'test']:
        # Obtain image and caption if in training or validation mode
        if split == 'train' or split == 'val':

            with h5py.File(os.path.join(output_dir, split+'/'+base_filename+'.hdf5'), 'a') as h:
                # load captions
                annotations_file = os.path.join(image_dir, 'annotations/captions_{}2017.json'.format(split))
                coco = COCO(annotations_file)
                ids = list(coco.anns.keys())

                # create dataset inside HDF5 file to store images
                images = h.create_dataset('coco', (len(ids), 3, 256, 256), dtype='uint8')
                encoded_captions = []
                captions_len = []

                print("\n Reading %s images and captions, storing to file... \n" % split)

                for i, index in enumerate(tqdm(ids)):
                    caption = coco.anns[index]["caption"]

                    img_id = coco.anns[index]["image_id"]
                    path = coco.loadImgs(img_id)[0]["file_name"]

                    # image = Image.open(os.path.join(image_dir+'images/{}/'.format(split), path))
                    # image = image.convert('RGB')
                    # image = image.resize((256, 256, 3), resample=Image.BILINEAR)
                    image = cv2.imread(os.path.join(image_dir + 'images/{}/'.format(split), path))
                    image = cv2.resize(image, dsize=(256, 256))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = image.reshape((3, 256, 256))
                    assert image.shape == (3, 256, 256)
                    assert np.max(image) <= 255

                    images[i] = image

                    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
                    caption = []
                    caption.append(vocab(vocab.start_word))
                    caption.extend(vocab(token) for token in tokens)
                    caption.append(vocab(vocab.end_word))
                    encoded_captions.append(caption)
                    captions_len.append(len(caption))

                assert images.shape[0]  == len(encoded_captions) == len(captions_len)

                # save encoded captions and their lengths to JSON files
                with open(os.path.join(output_dir, split+'/_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                    json.dump(encoded_captions, j)

                with open(os.path.join(output_dir, split+'/_CAPLENS_' + base_filename + '.json'), 'w') as j:
                    json.dump(captions_len, j)