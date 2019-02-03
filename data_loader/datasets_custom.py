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
sys.path.append('/Users/cuijingchen/Documents/Your Projects/I2T2I/data/coco/cocoapi/PythonAPI')
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
                 image_size,
                 batch_size,
                 transform,
                 vocab_threshold=5,
                 vocab_file="/Users/cuijingchen/Documents/Your Projects/I2T2I/data/coco/vocab.pkl",
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file="/Users/cuijingchen/Documents/Your Projects/I2T2I/data/coco/annotations/captions_train2017.json",
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



class Text2ImageDataset(Dataset):


    def __init__(self, datasetFile, which_set='train',transform=None):
        """
            @:param datasetFile (string): path for dataset file
            @:param which_set (string): "train:, "valid", "test"

        """
        self.datasetFile = datasetFile
        self.which_set = which_set
        assert self.which_set in {'train', 'valid', 'test'}

        self.transform = transform
        self.dataset = None
        self.dataset_keys = None
        self.h5py2int = lambda x: int(np.array(x))

    def __len__(self):
        f = h5py.File(self.datasetFile, 'r')
        self.dataset_keys = [str(k) for k in f[self.which_set].keys()]
        length = len(f[self.which_set])
        f.close()

        return length

    def __getitem__(self, idx):
        if self.dataset is None:
            self.dataset = h5py.File(self.datasetFile, mode='r')
            self.dataset_keys = [str(k) for k in self.dataset[self.which_set].keys()]

        example_name = self.dataset_keys[idx]
        example = self.dataset[self.which_set][example_name]

        # pdb.set_trace()

        right_image = bytes(np.array(example['img']))
        right_embed = np.array(example['embeddings'], dtype=float)
        wrong_image = bytes(np.array(self.find_wrong_image(example['class'])))
        wrong_embed = np.array(self.find_wrong_embed())

        right_image = Image.open(io.BytesIO(right_image)).convert("RGB")
        wrong_image = Image.open(io.BytesIO(wrong_image)).convert("RGB")
        right_image = self.transform(right_image)
        wrong_image = self.transform(wrong_image)

        #right_image = self.validate_image(right_image)
        #wrong_image = self.validate_image(wrong_image)

        txt = np.array(example['txt']).astype(str)

        sample = {
                'right_image': right_image,
                'right_embed': torch.FloatTensor(right_embed),
                'wrong_image': wrong_image,
                'wrong_embed': torch.FloatTensor(wrong_embed),
                'txt': str(txt)
                 }

        # sample['right_images'] = sample['right_images'].sub_(127.5).div_(127.5)
        # sample['wrong_images'] =sample['wrong_images'].sub_(127.5).div_(127.5)

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.which_set][example_name]
        _category = example['class']

        if _category != category:
            return example['img']

        return self.find_wrong_image(category)

    def find_wrong_embed(self):
        idx = np.random.randint(len(self.dataset_keys))
        example_name = self.dataset_keys[idx]
        example = self.dataset[self.which_set][example_name]
        return example['embeddings']