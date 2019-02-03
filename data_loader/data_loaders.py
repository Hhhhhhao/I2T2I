import torch
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from PIL import Image
import os

from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.datasets_custom import COCOCaptionDataset, Text2ImageDataset



def collate_fn(data):
    # sort the data in descentding order
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # merge images (from tuple of 1D tensor to 4D tensor)
    batch_images = torch.stack(images, 0)

    # merge captions (from tuple of 1D tensor to 2D tensor)
    batch_caption_lengths = [len(cap) for cap in captions]
    batch_captions = torch.zeros(len(captions), max(batch_caption_lengths)).long()
    for i, cap in enumerate(captions):
        end = batch_caption_lengths[i]
        batch_captions[i, :end] = cap[:end]

    return batch_images, batch_captions, batch_caption_lengths



class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class COCOCaptionDataLoader(DataLoader):
    """

    """
    def __init__(self, data_dir, which_set, image_size, batch_size, num_workers):

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}

        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms.ToTensor convert PIL images in range [0, 255] to a torch in range [0.0, 1.0]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.dataset = COCOCaptionDataset(self.data_dir, self.which_set, self.image_size, self.batch_size, self.transform)
        self.n_samples = len(self.dataset)

        if self.which_set == 'train':
            super(COCOCaptionDataLoader, self).__init__(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )
        else:
            super(COCOCaptionDataLoader, self).__init__(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn)


class Text2ImageDataLoader(DataLoader):
    def __init__(self, data_dir, dataset_name, which_set, image_size, batch_size, num_workers):
        """
            @:param dataset -- string: "birds" or "flowers"

        """

        self.data_dir = data_dir
        self.which_set = which_set
        assert self.which_set in {'train', 'val', 'test'}

        self.image_size = (image_size, image_size)
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms.ToTensor convert PIL images in range [0, 255] to a torch in range [0.0, 1.0]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # TODO: change mean and std
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        if dataset_name == "birds":
            self.dataset = Text2ImageDataset(os.path.join(self.data_dir, 'birds/birds.hdf5'), which_set=self.which_set, transform=self.transform)
        elif dataset_name == "flowers":
            self.dataset = Text2ImageDataset(os.path.join(self.data_dir, 'flowers/flowers.hdf5'), which_set=self.which_set, transform=self.transform)
        else:
            print("Dataset not supported, please select either birds or flowers.")
            exit()

        self.n_samples = len(self.dataset)

        if self.which_set == 'train':
            super(Text2ImageDataLoader, self).__init__(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers
            )
        else:
            super(Text2ImageDataLoader, self).__init__(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0)



if __name__ == '__main__':
    import nltk

    # data_loader = COCOCaptionDataLoader(
    #     data_dir='/Users/cuijingchen/Documents/Your Projects/I2T2I/data/coco/',
    #     which_set='train',
    #     image_size=128,
    #     batch_size=16,
    #     num_workers=0)

    bird_data_loader = Text2ImageDataLoader(
        data_dir='/Users/cuijingchen/Documents/Your Projects/I2T2I/data/', dataset_name = "flowers", which_set = "train", image_size = 64, batch_size = 16, num_workers=0
    )

    for i, sample in enumerate(bird_data_loader):
        print("done")
        print('images.shape:', sample['right_image'].shape)
        print('captions.shape:', sample['right_embed'].shape)
        print('images.shape:', sample['wrong_image'].shape)
        print('captions.shape:', sample['wrong_embed'].shape)
        print('txt:',sample['txt'])



