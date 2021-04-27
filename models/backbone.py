import torch
import torch.nn as nn
import torchvision

from torchvision.models.resnet import conv3x3
from torchvision.models.resnet import conv1x1
from torchvision.models.resnet import BasicBlock


def make_layer(block, planes, blocks, stride=1, dilate=False):
    norm_layer = self._norm_layer
    downsample = None
    previous_dilation = self.dilation

    if stride != 1 or self.inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            conv1x1(self.inplanes, planes * block.expansion, stride),
            norm_layer(planes * block.expansion),
        )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                        self.base_width, previous_dilation, norm_layer))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(block(self.inplanes, planes, groups=self.groups,
                            base_width=self.base_width, dilation=self.dilation,
                            norm_layer=norm_layer))

    return nn.Sequential(*layers)


class OCR34(nn.Module):
    def __init__(self):
        super(OCR34, self).__init__()


def make_downsample(ins, outs, stride):
    net = nn.Sequential(
        conv1x1(ins, outs, stride),
        nn.BatchNorm2d(outs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )

    return net


def make_ocr34():
    ocr34 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=(3, 3), bias=False),
        nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(inplace=True),

        nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 32),
            BasicBlock(32, 32)
        ),

        nn.Sequential(
            BasicBlock(32, 64, stride=2, downsample=make_downsample(32, 64, 2)),
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
