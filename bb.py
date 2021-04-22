import lmdb
import numpy as np
import cv2 as cv
from itertools import islice

import os
from os.path import sep as path_sep
from os.path import join as path_join

import torch
import torch.nn as nn
import torchvision

net = torchvision.models.resnet34(pretrained=False)
cnn = nn.Sequential(*(list(net.children())[: -2]))
# 修改第一层
cnn[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
cnn[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
cnn[6][0].conv1.stride = (2, 1)
cnn[6][0].downsample[0].stride = (2, 1)
cnn.add_module('avgPooling', nn.AvgPool2d(kernel_size=(4, 1), stride=1, padding=0))


x = torch.rand(1, 3, 32, 100)
# y = cnn[: 5](x)
# y = cnn[5](y)
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












