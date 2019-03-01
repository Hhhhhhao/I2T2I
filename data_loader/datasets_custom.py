import h5py
import json
import os
import io
import torch
import sys
import nltk
import numpy as np
dirname = os.path.dirname(__file__)
dirname = os.path.dirname(dirname)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/coco/cocoapi/PythonAPI'))
from pycocotools.coco import COCO
from tqdm import tqdm
from torch.utils.data import Dataset
from utils.data_processing import Vocabulary, COCOVocabulary, text_clean
from PIL import Image


class COCOCaptionDataset(Dataset):
    """
    A PyTorch MSCOCO Caption Dataset class to be used in a PyTorch COCO Caption DataLoader to create batches.
    """
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
                 vocab_from_file=False):

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}
        self.vocab = COCOVocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.transform = transform

        if self.which_set == 'train' or self.which_set == 'val':
            self.coco = COCO(os.path.join(data_dir, 'annotations/captions_{}2017.json'.format(which_set)))
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                          text_clean(str(self.coco.anns[self.ids[index]]["caption"])).lower())
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

            # image.show()

            image = self.transform(image)

            # Convert caption to tensor of word ids.
            caption = text_clean(caption)
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend(self.vocab(token) for token in tokens)
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            if self.which_set == 'train':
                return image, caption
            else:
                return img_id, image, caption

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


class CaptionDataset(Dataset):

    def __init__(self,
                 data_dir,
                 dataset_name,
                 which_set,
                 transform,
                 vocab_threshold=4,
                 start_word="<start>",
                 end_word="<end>",
                 unk_word="<unk>",
                 vocab_from_file=True
                 ):

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'valid', 'test'}

        self.vocab = Vocabulary(
            vocab_threshold=vocab_threshold,
            dataset_name=dataset_name,
            start_word=start_word,
            end_word=end_word,
            unk_word=unk_word,
            vocab_from_file=vocab_from_file,
            data_dir=data_dir)

        self.total_data = h5py.File(self.vocab.h5_file, mode='r')
        self.transform = transform

        self.data = self.total_data[which_set]
        self.ids = [str(k) for k in self.data.keys()]
        print("Obtaining caption lengths...")
        all_tokens = [nltk.tokenize.word_tokenize(
                      text_clean(str(np.array(self.data[index]['txt']))).lower())
                        for index in tqdm(self.ids)]
        self.caption_lengths = [len(token) for token in all_tokens]

    def __getitem__(self, index):
        # Obtain image and caption
        img_id = self.ids[index]
        caption = str(np.array(self.data[img_id]['txt']))
        image_path = bytes(np.array(self.data[img_id]['img']))
        image = Image.open(io.BytesIO(image_path)).convert("RGB")

        # image.show()

        image = self.transform(image)

        # Convert caption to tensor of word ids.
        caption = text_clean(caption)
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # tokens = [word for word in tokens]
        caption = []
        caption.append(self.vocab(self.vocab.start_word))
        caption.extend(self.vocab(token) for token in tokens)
        caption.append(self.vocab(self.vocab.end_word))
        caption = torch.Tensor(caption).long()

        if self.which_set == 'train' or self.which_set == 'valid':
            return image, caption
        else:
            return img_id, image, caption

    def __len__(self):
        return len(self.ids)


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

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}
        self.vocab = COCOVocabulary(vocab_threshold, vocab_file, start_word,
                                    end_word, unk_word, annotations_file, vocab_from_file)
        self.transform = transform

        if self.which_set == 'train' or self.which_set == 'val':
            self.coco = COCO(os.path.join(data_dir, 'annotations/captions_{}2017.json'.format(which_set)))
            self.ids = list(self.coco.anns.keys())
            print("Obtaining caption lengths...")
            all_tokens = [nltk.tokenize.word_tokenize(
                text_clean(str(self.coco.anns[self.ids[index]]["caption"])).lower())
                for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(os.path.join(data_dir, 'annotations/image_info_test2017')).read())
            self.paths = [item["file_name"] for item in test_info["images"]]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        ann_id = self.ids[index]
        z = self.coco.anns[ann_id]
        right_txt = self.coco.anns[ann_id]["caption"]
        right_txt = str(np.array(right_txt))
        img_id = self.coco.anns[ann_id]["image_id"]
        x = self.coco.loadImgs(img_id)[0]
        y = self.coco.loadCats(img_id)[0]
        right_image_path = self.coco.loadImgs(img_id)[0]["file_name"]


        wrong_txt = str(np.array(self.find_wrong_txt(self.data[img_id]['class'])))

        right_embed = np.array(self.data[img_id]['embeddings'], dtype=float)
        wrong_image_path = bytes(np.array(self.find_wrong_image(self.data[img_id]['class'])))
        wrong_embed = np.array(self.find_wrong_embed())

        # Processing images
        right_image = Image.open(io.BytesIO(right_image_path)).convert("RGB")
        wrong_image = Image.open(io.BytesIO(wrong_image_path)).convert("RGB")

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

    def find_wrong_txt(self, category):
        idx = np.random.randint(len(self.ids))
        img_id = self.ids[idx]

        _category = self.data[img_id]

        if _category != category:
            return self.data[img_id]['txt']

        return self.find_wrong_image(category)


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
        else:
            raise ValueError("data directory not found")

        self.which_set = which_set
        assert self.which_set in {'train', 'valid', 'test'}

        self.transform = transform
        self.total_data = h5py.File(self.h5_file, mode='r')
        self.data = self.total_data[which_set]
        self.ids = [str(k) for k in self.data.keys()]

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
                        for index in tqdm(self.ids)]
        self.caption_lengths = [len(token) for token in all_tokens]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        img_id = self.ids[index]
        right_txt = str(np.array(self.data[img_id]['txt']))
        wrong_txt = str(np.array(self.find_wrong_txt(self.data[img_id]['class'])))
        right_image_path = bytes(np.array(self.data[img_id]['img']))
        right_embed = np.array(self.data[img_id]['embeddings'], dtype=float)
        wrong_image_path = bytes(np.array(self.find_wrong_image(self.data[img_id]['class'])))
        wrong_embed = np.array(self.find_wrong_embed())

        # Processing images
        right_image = Image.open(io.BytesIO(right_image_path)).convert("RGB")
        wrong_image = Image.open(io.BytesIO(wrong_image_path)).convert("RGB")

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

    def find_wrong_txt(self, category):
        idx = np.random.randint(len(self.ids))
        img_id = self.ids[idx]

        _category = self.data[img_id]

        if _category != category:
            return self.data[img_id]['txt']

        return self.find_wrong_image(category)

    def compute_image_size(self):
        ids = []
        for i, id in enumerate(tqdm(self.ids)):

            if id[:-2] not in ids:
                ids.append(id[:-2])
        return len(ids)


if __name__ == '__main__':
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = COCOTextImageDataset(
        data_dir="/Users/leon/Projects/I2T2I/data/coco",
        which_set='val',
        transform=transform,
        vocab_threshold=4,
        start_word="<start>",
        end_word="<end>",
        unk_word="<unk>",
        vocab_from_file=False)

    # dataset = TextImageDataset(
    #     data_dir="/Users/leon/Projects/I2T2I/data/",
    #     dataset_name="flowers",
    #     which_set='test',
    #     transform=transform,
    #     vocab_threshold=4,
    #     start_word="<start>",
    #     end_word="<end>",
    #     unk_word="<unk>",
    #     vocab_from_file=False)
    dataset.__getitem__(2)
    print(len(dataset.vocab))
    print(dataset.vocab.word2idx)