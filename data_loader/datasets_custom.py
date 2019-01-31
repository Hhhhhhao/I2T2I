import h5py
import json
import os
import torch
import sys
import nltk
import numpy as np
sys.path.append('/Users/leon/Projects/I2T2I/data/coco/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_processing import Vocabulary
from PIL import Image


class COCOCaptionDataset(Dataset):
    """
    A PyTorch MSCOCO Caption Dataset class to be used in a PyTorch COCO Caption DataLoader to create batches.
    """
    def __init__(self,
                 data_dir,
                 which_set,
                 image_size,
                 batch_size,
                 transform,
                 vocab_threshold=5,
                 vocab_file="/Users/leon/Projects/I2T2I/data/coco/vocab.pkl",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file="/Users/leon/Projects/I2T2I/data/coco/annotations/captions_train2017.json",
                 vocab_from_file=True):

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}
        self.image_size = image_size
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.transform = transform

        if self.which_set == 'train' or self.which_set == 'val':
            self.coco = COCO(os.path.join(data_dir, 'annotations/captions_{}2017.json'.format(which_set)))
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                          str(self.coco.anns[self.ids[index]]["caption"]).lower())
                            for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(os.path.join(data_dir, 'annotations/image_info_test2017')).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __getitem__(self, index):
        # Obtain image and caption if in training or validation mode
        if self.which_set == 'train' or self.which_set == 'val':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]["caption"]
            img_id = self.coco.anns[ann_id]["image_id"]
            path = self.coco.loadImgs(img_id)[0]["file_name"]

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.data_dir + 'images/{}/'.format(self.which_set), path))
            image = image.convert("RGB")
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend(self.vocab(token) for token in tokens)
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            return image, caption

        # Obtain image, caption in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-precess using transform
            image = Image.open(os.path.join(self.data_dir + 'images/test/', path))
            image = image.convert("RGB")
            orig_image = np.array(image)
            image = self.transform(image)

            return orig_image, image

    def __len__(self):
        if self.which_set == "train" or self.which_set == "val":
            return len(self.ids)
        else:
            return len(self.paths)


