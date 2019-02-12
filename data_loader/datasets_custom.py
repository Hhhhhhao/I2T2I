import h5py
import json
import os
import io
import torch
import sys
import nltk
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.append('/home/s1784380/I2T2I/data/coco/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from tqdm import tqdm
from utils.data_processing import Vocabulary




class COCOCaptionDataset(Dataset):
    """
    A PyTorch MSCOCO Caption Dataset class to be used in a PyTorch COCO Caption DataLoader to create batches.
    """
    def __init__(self,
                 data_dir,
                 which_set,
                 transform,
                 vocab_threshold=5,
                 vocab_file="/home/s1784380/I2T2I/data/coco/vocab.pkl",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file="/home/s1784380/I2T2I/data/coco/annotations/captions_train2017.json",
                 vocab_from_file=True):

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}
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


class Text2ImageDataset(Dataset):

    def __init__(self, data_dir, dataset_name, which_set='train', transform=None):
        """
            @:param datasetFile (string): path for dataset file
            @:param which_set (string): "train", "valid", "test"

        """

        if os.path.exists(data_dir):
            assert dataset_name in {'birds', 'flowers'}, "wrong dataset name"
            self.h5_file = os.path.join(data_dir, '{}/{}.hdf5'.format(dataset_name, dataset_name))
        else:
            raise ValueError("data directory not found")

        self.which_set = which_set
        assert self.which_set in {'train', 'valid', 'test'}

        self.transform = transform
        self.total_data = h5py.File(self.h5_file, mode='r')
        self.data = self.total_data[self.which_set]
        self.ids = [str(k) for k in self.data.keys()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        img_id = self.ids[index]
        caption = str(np.array(self.data[img_id]['txt']))
        right_image_path = bytes(np.array(self.data[img_id]['img']))
        right_embed = np.array(self.data[img_id]['embeddings'], dtype=float)
        wrong_image_path = bytes(np.array(self.find_wrong_image(self.data[img_id]['class'])))
        wrong_embed = np.array(self.find_wrong_embed())

        right_image = Image.open(io.BytesIO(right_image_path)).convert("RGB")
        wrong_image = Image.open(io.BytesIO(wrong_image_path)).convert("RGB")
        right_image = self.transform(right_image)
        wrong_image = self.transform(wrong_image)

        sample = {
                'right_image': right_image,
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_image': wrong_image,
                'wrong_embed': torch.FloatTensor(wrong_embed),
                'txt': caption
                 }

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.ids))
        img_id = self.ids[idx]
        _category = self.data[img_id]

        if _category != category:
            return self.data[img_id]['img']

        return self.find_wrong_image(category)

    def find_wrong_embed(self):
        idx = np.random.randint(len(self.ids))
        img_id = self.ids[idx]
        return self.data[img_id]['embeddings']