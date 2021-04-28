#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
from PIL import Image
import numpy as np
import imgaug.augmenters as iaa


class Dataset_lmdb(Dataset):

    def __init__(self, root=None, in_channels=3, transform=None, target_transform=None):
        self.in_channels = in_channels
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            # 转换成单通道
            if self.in_channels == 1:
                img = img.convert('L')
            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class RandomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples

import cv2 as cv
class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, augment=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.augment = augment
        # ### 数据增强
        if augment is True:
            # 注意 还不可以使用， 需要重新设定增强函数的数值
            self.augmentor = iaa.Sequential([
                # iaa.Crop(percent=(0.0, 0.02), keep_size=True),
                iaa.GaussianBlur(sigma=(0, 0.2)),
                iaa.MultiplyHueAndSaturation((0.05, 0.2), per_channel=True),
                # iaa.GammaContrast((0.7, 1)),
                iaa.PiecewiseAffine(scale=(0.01, 0.05))
            ])


    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = ResizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        # 数据增强
        if self.augment is True:
            images = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
            images = self.augmentor(images=images).astype(np.float32) / 255
            images = torch.from_numpy(images).permute(0, 3, 1, 2).detach()

        return images, labels


    def show(self, images):

        for i in range(10):
            a = images[i, ...].permute(1, 2, 0)
            a = a.numpy()
            cv.cvtColor(a, cv.COLOR_RGB2BGR)
            cv.imshow('image', a), cv.waitKeyEx()

        cv.destroyAllWindows()

