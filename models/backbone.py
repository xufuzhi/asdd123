import torch
import torch.nn as nn
import torchvision

from torchvision.models.resnet import conv3x3
from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import BasicBlock


def make_downsample(in_channels, outs, stride):
    net = nn.Sequential(
        conv1x1(in_channels, outs, stride),
        nn.BatchNorm2d(outs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

    return net


def make_ocr34(in_channels=3):
    ocr34 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=(3, 3), bias=False),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),

        nn.Sequential(
            BasicBlock(32, 64, stride=1, downsample=make_downsample(32, 64, 1)),
            BasicBlock(64, 64)
        ),

        nn.Sequential(
            BasicBlock(64, 64, stride=2, downsample=make_downsample(64, 64, 2)),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        ),

        nn.Sequential(
            BasicBlock(64, 128, stride=(2, 1), downsample=make_downsample(64, 128, (2, 1))),
            BasicBlock(128, 128),
            BasicBlock(128, 128)
        ),

        nn.Sequential(
            BasicBlock(128, 256, stride=(2, 1), downsample=make_downsample(128, 256, (2, 1))),
            BasicBlock(256, 256),
            BasicBlock(256, 256),
            BasicBlock(256, 256)
        ),

        nn.Sequential(
            BasicBlock(256, 512, stride=(2, 1), downsample=make_downsample(256, 512, (2, 1))),
            BasicBlock(512, 512),
            BasicBlock(512, 512)
        ),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )

    return ocr34


def make_ocr7(in_channels, leaky_relu=False):
    ks = [3, 3, 3, 3, 3, 3, 2]
    ps = [1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1]
    nm = [64, 128, 256, 256, 512, 512, 512]
    # nm = [64, 128, 128, 128, 256, 256, 512]

    def conv_relu(i, batch_normalization=False):
        nIn = in_channels if i == 0 else nm[i - 1]
        nOut = nm[i]
        cnn.add_module('conv{0}'.format(i),
                       nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        if batch_normalization:
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        if leaky_relu:
            cnn.add_module('relu{0}'.format(i),
                           nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    cnn = nn.Sequential()
    conv_relu(0, True)
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d([2, 2], [2, 1]))  # 64x16x64
    conv_relu(1)
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
    conv_relu(2, True)
    conv_relu(3)
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x32
    conv_relu(4, True)
    conv_relu(5)
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 2), (0, 1)))  # 512x2x32
    conv_relu(6, True)

    return cnn


def make_ocr10(in_channels, leaky_relu=False):
    ks = [7, 3, 3, 3, 3, 3, 3, 3, 3, 2]
    ps = [3, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    ss = [1, 1, 1, 1, 1, 1, 1, 1, 1, (2, 1)]
    nm = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512]
    # nm = [64, 128, 128, 128, 256, 256, 512]

    def conv_relu(i, batch_normalization=False):
        nIn = in_channels if i == 0 else nm[i - 1]
        nOut = nm[i]
        cnn.add_module('conv{0}'.format(i),
                       nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
        if batch_normalization:
            cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
        if leaky_relu:
            cnn.add_module('relu{0}'.format(i),
                           nn.LeakyReLU(0.2, inplace=True))
        else:
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

    cnn = nn.Sequential()
    conv_relu(0)
    conv_relu(1)
    conv_relu(2, True)
    cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2), (2, 2)))
    conv_relu(3)
    conv_relu(4, True)
    cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2), (2, 1)))
    conv_relu(5)
    conv_relu(6, True)
    cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
    conv_relu(7)  # 512x1x32
    conv_relu(8, True)
    cnn.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 2), (0, 1)))
    conv_relu(9)

    return cnn


def make_res_pp(in_channels=3):
    net = torchvision.models.resnet34(pretrained=False)
    l1 = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
    )
    cnn = nn.Sequential(*(list(net.children())[: -2]))
    # 修改网络层
    cnn[0] = l1
    cnn[5][0].conv1.stride = (2, 1)
    cnn[5][0].downsample[0].stride = (2, 1)
    cnn[6][0].conv1.stride = (2, 1)
    cnn[6][0].downsample[0].stride = (2, 1)
    cnn[7][0].conv1.stride = (2, 1)
    cnn[7][0].downsample[0].stride = (2, 1)
    cnn.add_module('maxPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    return cnn



