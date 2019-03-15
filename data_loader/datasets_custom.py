import h5py
import json
import os
import io
import torch
import sys
import nltk
import numpy as np
import pandas as pd
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/coco/cocoapi/PythonAPI'))
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_processing import Vocabulary, COCOVocabulary, text_clean
from PIL import Image
from collections import OrderedDict


class COCOTextImageDataset(Dataset):
    def __init__(self,
                 data_dir,
                 which_set,
                 transform,
                 vocab_threshold=4,
                 vocab_file=os.path.join(dirname, "data/coco/vocab.pkl"),
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 annotations_file=os.path.join(dirname, "data/coco/annotations/captions_train2017.json"),
                 vocab_from_file=True
                 ):
        """
            @:param datasetFile (string): path for dataset file
            @:param which_set (string): "train:, "valid", "test"
        """
        if data_dir[-1] != '/':
            data_dir += '/'
            
        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}
        self.vocab = COCOVocabulary(vocab_threshold, vocab_file, start_word,
                                    end_word, unk_word, annotations_file, vocab_from_file)
        self.transform = transform

        if self.which_set == 'train' or self.which_set == 'val':
            self.coco = COCO(os.path.join(data_dir, 'annotations/captions_{}2017.json'.format(which_set)))
            self.ann_ids = list(self.coco.anns.keys())

            self.classes = OrderedDict()
            class_cnt = 0
            for ann_id in self.ann_ids:
                img_id = self.coco.anns[ann_id]["image_id"]
                if img_id not in self.classes.keys():
                    class_cnt += 1
                    self.classes[img_id] = class_cnt

            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                text_clean(str(self.coco.anns[self.ann_ids[index]]["caption"])).lower())
                for index in tqdm(np.arange(len(self.ann_ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(os.path.join(data_dir, 'annotations/image_info_test2017')).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __len__(self):
        return len(self.ann_ids)

    def __getitem__(self, index):
        right_ann_id = self.ann_ids[index]

        right_txt = self.coco.anns[right_ann_id]["caption"]
        right_txt = str(np.array(right_txt))
        right_img_id = self.coco.anns[right_ann_id]["image_id"]
        right_image_path = self.coco.loadImgs(right_img_id)[0]["file_name"]
        class_id = self.classes[right_img_id]
        # TODO use DAMSM model to get embedding
        right_embed = [0]

        wrong_img_id = self.find_wrong_img_id(right_img_id)
        wrong_txt = str(np.array(self.find_wrond_txt(right_img_id)))
        wrong_image_path = self.coco.loadImgs(wrong_img_id)[0]["file_name"]
        # TODO use DAMSM model to get embedding
        wrong_embed = [0]

        # Processing images
        right_image = Image.open(os.path.join(self.data_dir + 'images/{}/'.format(self.which_set), right_image_path))
        right_image = right_image.convert("RGB")
        wrong_image = Image.open(os.path.join(self.data_dir + 'images/{}/'.format(self.which_set), wrong_image_path))
        wrong_image = wrong_image.convert("RGB")

        right_image_32 = right_image.resize((32, 32))
        wrong_image_32 = wrong_image.resize((32, 32))
        right_image_64 = right_image.resize((64, 64))
        wrong_image_64 = wrong_image.resize((64, 64))
        right_image_128 = right_image.resize((128, 128))
        wrong_image_128 = wrong_image.resize((128, 128))
        right_image_256 = right_image.resize((256, 256))
        wrong_image_256 = wrong_image.resize((256, 256))

        right_image_32 = self.transform(right_image_32)
        wrong_image_32 = self.transform(wrong_image_32)
        right_image_64 = self.transform(right_image_64)
        wrong_image_64 = self.transform(wrong_image_64)
        right_image_128 = self.transform(right_image_128)
        wrong_image_128 = self.transform(wrong_image_128)
        right_image_256 = self.transform(right_image_256)
        wrong_image_256 = self.transform(wrong_image_256)

        # Processing txt
        # Convert caption to tensor of word ids.
        right_txt = text_clean(right_txt)
        right_tokens = nltk.tokenize.word_tokenize(str(right_txt).lower())
        right_caption = []
        right_caption.append(self.vocab(self.vocab.start_word))
        right_caption.extend(self.vocab(token) for token in right_tokens)
        right_caption.append(self.vocab(self.vocab.end_word))
        right_caption = torch.Tensor(right_caption).long()

        wrong_txt = text_clean(wrong_txt)
        wrong_tokens = nltk.tokenize.word_tokenize(str(wrong_txt).lower())
        wrong_caption = []
        wrong_caption.append(self.vocab(self.vocab.start_word))
        wrong_caption.extend(self.vocab(token) for token in wrong_tokens)
        wrong_caption.append(self.vocab(self.vocab.end_word))
        wrong_caption = torch.Tensor(wrong_caption).long()

        sample = {
                'right_img_id': right_img_id,
                'right_class_id': class_id,
                'right_image_32': right_image_32,
                'right_image_64': right_image_64,
                'right_image_128': right_image_128,
                'right_image_256': right_image_256,
                'right_embed': torch.FloatTensor(right_embed),
                'right_caption': right_caption,
                'right_txt': right_txt,
                'wrong_image_32': wrong_image_32,
                'wrong_image_64': wrong_image_64,
                'wrong_image_128': wrong_image_128,
                'wrong_image_256': wrong_image_256,
                'wrong_embed': torch.FloatTensor(wrong_embed),
                'wrong_caption': wrong_caption,
                'wrong_txt': wrong_txt,
                 }

        return sample

    def find_wrong_img_id(self, right_img_id):
        idx = np.random.randint(len(self.ann_ids))
        ann_id = self.ann_ids[idx]
        img_id = self.coco.anns[ann_id]["image_id"]

        if img_id != right_img_id:
            return img_id

        return self.find_wrong_img_id(right_img_id)

    def find_wrond_txt(self, right_img_id):
        idx = np.random.randint(len(self.ann_ids))
        ann_id = self.ann_ids[idx]
        img_id = self.coco.anns[ann_id]["image_id"]

        if img_id != right_img_id:
            txt = self.coco.anns[ann_id]["caption"]
            return txt

        return self.find_wrond_txt(right_img_id)


class TextImageDataset(Dataset):
    def __init__(self,
                 data_dir,
                 dataset_name,
                 which_set='train',
                 transform=None,
                 vocab_threshold=4,
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 vocab_from_file=False
                 ):
        """
            @:param datasetFile (string): path for dataset file
            @:param which_set (string): "train:, "valid", "test"
        """

        if os.path.exists(data_dir):
            assert dataset_name in {'birds', 'flowers'}, "wrong dataset name"
            self.h5_file = os.path.join(data_dir, '{}/{}.hdf5'.format(dataset_name, dataset_name))
            self.data_dir = data_dir
            self.dataset_name = dataset_name
        else:
            raise ValueError("data directory not found")

        self.which_set = which_set
        assert self.which_set in {'train', 'valid', 'test'}

        self.transform = transform
        self.total_data = h5py.File(self.h5_file, mode='r')
        self.data = self.total_data[which_set]
        self.img_ids = [str(k) for k in self.data.keys()]

        if dataset_name == 'birds':
            # load bounding box
            self.bbox = self.load_bounding_box()
            # load class file
            class_file = os.path.join(data_dir, dataset_name, 'CUB_200_2011', 'classes.txt')
            self.classes = OrderedDict()
            with open(class_file, 'rb') as f:
                for line in f:
                    (key, val) = line.split()
                    self.classes[val.decode("utf-8")] = int(key)
        elif dataset_name == 'flowers':
            self.bbox = None
            class_file = os.path.join(data_dir, dataset_name, 'classes.txt')
            self.classes = OrderedDict()
            with open(class_file, 'rb') as f:
                for i, line in enumerate(f):
                    val = line.split()[0]
                    self.classes[val.decode("utf-8")] = i+1

        self.vocab = Vocabulary(
            vocab_threshold=vocab_threshold,
            dataset_name=dataset_name,
            start_word=start_word,
            end_word=end_word,
            unk_word=unk_word,
            vocab_from_file=vocab_from_file,
            data_dir=data_dir)

        all_tokens = [nltk.tokenize.word_tokenize(
                      text_clean(str(np.array(self.data[index]['txt']))).lower())
                        for index in tqdm(self.img_ids)]
        self.caption_lengths = [len(token) for token in all_tokens]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        img_id = self.img_ids[index]
        class_name = str(np.array(self.data[img_id]['class']))
        class_id = self.classes[class_name]

        right_txt = str(np.array(self.data[img_id]['txt']))
        wrong_txt = str(np.array(self.find_wrong_txt(self.data[img_id]['class'])))
        right_image_path = bytes(np.array(self.data[img_id]['img']))
        right_embed = np.array(self.data[img_id]['embeddings'], dtype=float)
        wrong_image_path, wrong_img_id = self.find_wrong_image(self.data[img_id]['class'])
        wrong_image_path = bytes(np.array(wrong_image_path))
        wrong_embed = np.array(self.find_wrong_embed())

        # Processing images
        right_image = Image.open(io.BytesIO(right_image_path)).convert("RGB")
        wrong_image = Image.open(io.BytesIO(wrong_image_path)).convert("RGB")

        if self.bbox is not None:
            right_image_bbox = self.bbox[str(np.array(self.data[img_id]["name"]))]
            wrong_image_bbox = self.bbox[str(np.array(self.data[wrong_img_id]["name"]))]
            right_image = self.crop_image(right_image, right_image_bbox)
            wrong_image = self.crop_image(wrong_image, wrong_image_bbox)

        right_image_32 = right_image.resize((32, 32))
        wrong_image_32 = wrong_image.resize((32, 32))
        right_image_64 = right_image.resize((64, 64))
        wrong_image_64 = wrong_image.resize((64, 64))
        right_image_128 = right_image.resize((128, 128))
        wrong_image_128 = wrong_image.resize((128, 128))
        right_image_256 = right_image.resize((256, 256))
        wrong_image_256 = wrong_image.resize((256, 256))

        right_image_32 = self.transform(right_image_32)
        wrong_image_32 = self.transform(wrong_image_32)
        right_image_64 = self.transform(right_image_64)
        wrong_image_64 = self.transform(wrong_image_64)
        right_image_128 = self.transform(right_image_128)
        wrong_image_128 = self.transform(wrong_image_128)
        right_image_256 = self.transform(right_image_256)
        wrong_image_256 = self.transform(wrong_image_256)

        # Processing txt
        # Convert caption to tensor of word ids.
        right_txt = text_clean(right_txt)
        right_tokens = nltk.tokenize.word_tokenize(str(right_txt).lower())
        right_caption = []
        right_caption.append(self.vocab(self.vocab.start_word))
        right_caption.extend(self.vocab(token) for token in right_tokens)
        right_caption.append(self.vocab(self.vocab.end_word))
        right_caption = torch.Tensor(right_caption).long()

        wrong_txt = text_clean(wrong_txt)
        wrong_tokens = nltk.tokenize.word_tokenize(str(wrong_txt).lower())
        wrong_caption = []
        wrong_caption.append(self.vocab(self.vocab.start_word))
        wrong_caption.extend(self.vocab(token) for token in wrong_tokens)
        wrong_caption.append(self.vocab(self.vocab.end_word))
        wrong_caption = torch.Tensor(wrong_caption).long()

        sample = {
                'right_img_id': img_id,
                'right_class_id': class_id,
                'right_image_32': right_image_32,
                'right_image_64': right_image_64,
                'right_image_128': right_image_128,
                'right_image_256': right_image_256,
                'right_embed': torch.FloatTensor(right_embed),
                'right_caption': right_caption,
                'right_txt': right_txt,
                'wrong_image_32': wrong_image_32,
                'wrong_image_64': wrong_image_64,
                'wrong_image_128': wrong_image_128,
                'wrong_image_256': wrong_image_256,
                'wrong_embed': torch.FloatTensor(wrong_embed),
                'wrong_caption': wrong_caption,
                'wrong_txt': wrong_txt,
                 }

        return sample

    def find_wrong_image(self, category):
        idx = np.random.randint(len(self.img_ids))
        img_id = self.img_ids[idx]
        _category = self.data[img_id]

        if _category != category:
            return self.data[img_id]['img'], img_id

        return self.find_wrong_image(category)

    def find_wrong_embed(self):
        idx = np.random.randint(len(self.img_ids))
        img_id = self.img_ids[idx]
        return self.data[img_id]['embeddings']

    def find_wrong_txt(self, category):
        idx = np.random.randint(len(self.img_ids))
        img_id = self.img_ids[idx]

        _category = self.data[img_id]

        if _category != category:
            return self.data[img_id]['txt']

        return self.find_wrong_image(category)

    def load_bounding_box(self):
        bbox_path = os.path.join(self.data_dir, self.dataset_name, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(self.data_dir, self.dataset_name, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()

        # processing filenames to match data example name
        for i in range(len(filenames)):
            filename = filenames[i][:-4]
            filename = filename.split('/')
            filename = filename[1]
            filenames[i] = filename
        print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()

            key = filenames[i]
            filename_bbox[key] = bbox
        #
        return filename_bbox

    def crop_image(self, image, bbox):
        width, height = image.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        image = image.crop([x1, y1, x2, y2])
        return image


if __name__ == '__main__':
    dataset = TextImageDataset(
        data_dir='/Users/leon/Projects/I2T2I/data/',
        dataset_name='birds',
        which_set='train',
    )

    # dataset = COCOTextImageDataset(
    #     data_dir='/Users/leon/Projects/I2T2I/data/coco/',
    #     which_set='val',
    #     transform=None
    # )

    sample = dataset.__getitem__(1)
