# -*- coding: utf-8 -*-
import os
import torch
from torch.autograd import Variable



import numpy as np
import scipy
import matplotlib.pyplot as plt
import cv2
import scipy.ndimage
import shutil
import scipy.misc as misc
from PIL import Image


def mkdirs(folders, erase=False):
    if type(folders) is not list:
        folders = [folders]
    for fold in folders:
        if not os.path.exists(fold):
            os.makedirs(fold)
        else:
            if erase:
                shutil.rmtree(fold)
                os.makedirs(fold)


def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_) / (max_ - min_ + 1e-9)
    X = X * 255
    return X.astype(np.uint8)


def imread(imgfile):
    assert os.path.exists(imgfile), '{} does not exist!'.format(imgfile)
    srcBGR = cv2.imread(imgfile)
    destRGB = cv2.cvtColor(srcBGR, cv2.COLOR_BGR2RGB)
    return destRGB


def writeImg(array, savepath):
    scipy.misc.imsave(savepath, array)


def imresize(img, resizeratio=1):
    '''Take care of cv2 reshape squeeze behevaior'''
    if resizeratio == 1:
        return img
    outshape = (int(img.shape[1] * resizeratio), int(img.shape[0] * resizeratio))
    # temp = cv2.resize(img, outshape).astype(float)
    temp = misc.imresize(img, size=outshape).astype(float)
    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imresize_shape(img, outshape):
    if len(img.shape) == 3:
        if img.shape[0] == 1 or img.shape[0] == 3:
            transpose_img = np.transpose(img, (1, 2, 0))
            _img = imresize_shape(transpose_img, outshape)
            return np.transpose(_img, (2, 0, 1))
    if len(img.shape) == 4:
        img_out = []
        for this_img in img:
            img_out.append(imresize_shape(this_img, outshape))
        return np.stack(img_out, axis=0)

    img = img.astype(np.float32)
    outshape = (int(outshape[1]), int(outshape[0]))

    # temp = cv2.resize(img, outshape).astype(float)
    temp = misc.imresize(img, size=outshape, interp='bilinear').astype(float)

    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp


def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize=size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()


def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1) // int(batch_size)  # the floor
    # Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)

    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx * batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd


def IndexH5(h5_array, indices):
    read_list = []
    for idx in indices:
        read_list.append(h5_array[idx])
    return np.stack(read_list, 0)


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

def get_end_symbol_index(caption_list):
    if 1 in caption_list:
        return caption_list.index(1) + 1
    else:
        return len(caption_list)

def get_caption_lengths(captions_list):
    caption_lengths = [get_end_symbol_index(caption) for caption in captions_list]
    caption_lengths.sort(reverse=True)
    batch_captions = torch.zeros(len(captions_list), max(caption_lengths)).long()
    for i, cap in enumerate(captions_list):
        end = caption_lengths[i]
        batch_captions[i, :end] = torch.tensor(cap[:end]).long()

    if torch.cuda.is_available():
        batch_captions = batch_captions.cuda()

    return batch_captions, caption_lengths


def to_numpy(src):
    if type(src) == np.ndarray:
        return src
    elif type(src) == Variable:
        x = src.data
    else:
        x = src
    return x.detach().cpu().numpy()


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)
    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk
    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)