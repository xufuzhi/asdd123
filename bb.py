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

import copy
net = torchvision.models.resnet50(pretrained=False)
new_1 = copy.deepcopy(net.layer1[0])
new_1.conv1 = nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
new_1.bn1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
new_1.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
new_1.bn2 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
new_1.conv3 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
new_1.bn3 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
new_1.downsample[0] = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
new_1.downsample[1] = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

l1 = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
    nn.ReLU(inplace=True),
    new_1
)

cnn = nn.Sequential(l1, *(list(net.children())[3: -2]))
# 修改网络层
cnn[3][0].conv2.stride = (2, 1)
cnn[3][0].downsample[0].stride = (2, 1)
cnn[4][0].conv2.stride = (2, 1)
cnn[4][0].downsample[0].stride = (2, 1)
cnn[5][0].conv2.stride = (2, 1)
cnn[5][0].downsample[0].stride = (2, 1)
cnn.add_module('maxPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

x = torch.rand(1, 3, 32, 128)
y = cnn(x)




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












