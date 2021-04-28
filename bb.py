import lmdb
import numpy as np
import cv2 as cv
from itertools import islice
import time

import os
from os.path import sep as path_sep
from os.path import join as path_join

import torch
import torch.nn as nn
import torchvision


path = '/home/xfz/Projects/PycharmProjects/TextRecognitionDataGenerator-master/trdg/out'
f_nameList = os.listdir(path)
for n in f_nameList:
    n_ = n.replace(' ', '')
    n = os.path.join(path, n)
    n_ = os.path.join(path, n_)
    os.rename(n, n_)
    print(n, '===>', n_)


###################################################
import models.crnn as crnn

class print_time():
    t = 0

    @classmethod
    def strat(cls):
        torch.cuda.synchronize()
        cls.t = time.time()

    @classmethod
    def end(cls):
        torch.cuda.synchronize()
        print('use time: ', time.time() - cls.t)

    # 读取字母表
with open('data/en.alphabet', encoding='utf-8') as f:
    alphabet = f.read().strip()

net_crnn = crnn.CRNN_ocr34(32, 3, len(alphabet) + 1, 256, d_bug='maxpool', rudc=False).to('cuda').eval()
# net_crnn = crnn.CRNN(32, 3, len(alphabet) + 1, 256).to('cuda').eval()

net_crnn = net_crnn.cnn
x = torch.rand([100, 3, 32, 128]).cuda()

with torch.no_grad():
    y = net_crnn(x)

    torch.cuda.synchronize()
    t1 = time.time()
    for _ in range(10):
        y = net_crnn(x)
    torch.cuda.synchronize()
    print('time: ', time.time() - t1)

raise ValueError

##################################################################



filepath = './data/lmdb_5w'
# filepath = '../../datas/aug240w'

outroot = path_join(*filepath.rsplit(path_sep, 1)) + '_img'
outroot = outroot.replace('lmdb_5w', 'lmdb_2w')
# assert not os.path.exists(outroot)
# os.makedirs(outroot)
# os.makedirs(path_join(outroot, 'images'))
# ### 读取LMDB数据集中图片并显示出来，验证一下数据集是否制作成功
val_num = 10
with lmdb.open(filepath) as env, open(path_join(outroot, 'train.txt'), 'w', encoding='utf-8') as f:
    txn = env.begin()
    # for key, value in islice(txn.cursor(), val_num):
    for i, (key, value) in enumerate(txn.cursor(), start=30000):
        imageBuf = np.fromstring(value, dtype=np.uint8)
        img = cv.imdecode(imageBuf, cv.IMREAD_GRAYSCALE)
        if img is not None:
            # 得到图片对应 label
            key_ = key.decode().replace('image', 'label', 1).encode()
            label = txn.get(key_).decode()
            ################################
            # 保存图片
            cv.imwrite(path_join(outroot, 'images', str(i)) + '.png', img)
            # 保存label
            l = str(i) + '.png' + ' ' + label
            f.writelines(l + '\n')
            #################################
            # print(label)
            # 显示图片
            # cv.imshow('image', img)
            # cv.waitKey()
        else:  # 标签数据，不处理
            pass
            # print('key: %s    label: %s' % (key, value))












