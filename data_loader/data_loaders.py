import torch
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.datasets_custom import COCOCaptionDataset


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


if __name__ == '__main__':
    import nltk

    data_loader = COCOCaptionDataLoader(
        data_dir='/Users/leon/Projects/I2T2I/data/coco/',
        which_set='train',
        image_size=128,
        batch_size=16,
        num_workers=0)

    sample_caption = 'A person doing a trick on a rail while riding a skateboard.'
    sample_tokens = nltk.tokenize.word_tokenize(str(sample_caption).lower())
    print(sample_tokens)

    sample_caption = []
    start_word = data_loader.dataset.vocab.start_word
    print('Special start word:', start_word)
    sample_caption.append(data_loader.dataset.vocab(start_word))
    print(sample_caption)

    sample_caption.extend([data_loader.dataset.vocab(token) for token in sample_tokens])
    print(sample_caption)

    end_word = data_loader.dataset.vocab.end_word
    print('Special end word:', end_word)

    sample_caption.append(data_loader.dataset.vocab(end_word))
    print(sample_caption)

    sample_caption = torch.Tensor(sample_caption).long()
    print(sample_caption)

    # Preview the word2idx dictionary.
    print(dict(list(data_loader.dataset.vocab.word2idx.items())[:10]))

    # Print the total number of keys in the word2idx dictionary.
    print('Total number of tokens in vocabulary:', len(data_loader.dataset.vocab))

    for i, (images, captions, caption_lengths) in enumerate(data_loader):
        print("done")

    print('images.shape:', images.shape)
    print('captions.shape:', captions.shape)

    # Print the pre-processed images and captions.
    print('images:', images)
    print('captions:', captions)



        