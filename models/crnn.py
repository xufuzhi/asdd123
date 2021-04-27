import time

import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        # nm = [64, 128, 128, 128, 256, 256, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x32
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x32
        convRelu(6, True)  # 512x1x32

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output




######################################################################################################
class CRNN_res(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug = 'maxpool', rudc=True):
        super(CRNN_res, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        print('CRNN_res')

        net = torchvision.models.resnet34(pretrained=False)
        cnn = nn.Sequential(*(list(net.children())[: -2]))
        # 修改网络层
        if rudc:
            cnn[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            print('============> rudc')
        else:
            cnn[0].stride = (1, 1)
        cnn[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        cnn[6][0].conv1.stride = (2, 1)
        cnn[6][0].downsample[0].stride = (2, 1)
        if d_bug == 'avgpool':
            cnn.add_module('avgPooling', nn.AvgPool2d(kernel_size=(4, 1), stride=1, padding=0))
            print('==============> avgPooling')
        elif d_bug == 'maxpool':
            cnn.add_module('avgPooling', nn.MaxPool2d(kernel_size=(4, 1), stride=1, padding=0))
            print('==============> maxpool')
        else:
            cnn.add_module('last_conv', nn.Conv2d(512, 512, kernel_size=4, stride=(4, 1), padding=0, bias=False))
            cnn.add_module('last_bn', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            cnn.add_module('last_relu', nn.ReLU(inplace=True))
            print('==============> conv')

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )



    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output



###########################################################################################################
class CRNN_res_1(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug='maxpool', rudc=True):
        print('CRNN_res_1')
        super(CRNN_res_1, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        net = torchvision.models.resnet34(pretrained=False)
        cnn = nn.Sequential(*(list(net.children())[: -2]))
        # 修改网络层
        if rudc:
            cnn[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            cnn[3] = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
            print('============> rudc')
        else:
            cnn[0].stride = (1, 1)

        cnn[5][0].conv1.stride = (2, 1)
        cnn[5][0].downsample[0].stride = (2, 1)
        cnn[6][0].conv1.stride = (2, 1)
        cnn[6][0].downsample[0].stride = (2, 1)
        cnn[7][0].conv1.stride = (2, 1)
        cnn[7][0].downsample[0].stride = (2, 1)
        if d_bug == 'avgpool':
            cnn.add_module('avgPooling', nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0))
            print('==============> avgPooling')
        elif d_bug == 'maxpool':
            cnn.add_module('avgPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            print('==============> maxpool')
        else:
            cnn.add_module('last_conv', nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False))
            cnn.add_module('last_bn',
                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            cnn.add_module('last_relu', nn.ReLU(inplace=True))
            print('==============> conv')

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output

###########################################################################################################
class CRNN_res_pp(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug='maxpool', rudc=True):
        print('CRNN_res_pp')
        super(CRNN_res_pp, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        net = torchvision.models.resnet34(pretrained=False)
        l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
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
        if d_bug == 'avgpool':
            cnn.add_module('avgPooling', nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0))
            print('==============> avgPooling')
        elif d_bug == 'maxpool':
            cnn.add_module('maxPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            print('==============> maxpool')
        else:
            cnn.add_module('last_conv', nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0, bias=False))
            cnn.add_module('last_bn',
                           nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            cnn.add_module('last_relu', nn.ReLU(inplace=True))
            print('==============> conv')

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output


###########################################################################################################
class CRNN_vgg16(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug='maxpool', rudc=True):
        print('CRNN_vgg16')
        super(CRNN_vgg16, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        net = torchvision.models.vgg16_bn(pretrained=False)
        cnn = net.features
        cnn[6] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
        cnn[23] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)
        cnn[33] = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0, dilation=1, ceil_mode=False)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output


######################################################################################################
import copy
class CRNN_m(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug = 'maxpool', rudc=True):
        super(CRNN_m, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        print('CRNN_m')

        net = torchvision.models.resnet34(pretrained=False)
        cnn = nn.Sequential(*(list(net.children())[: -2]))
        # 修改网络层
        if rudc:
            cnn[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            print('============> rudc')
        else:
            cnn[0].stride = (1, 1)
        cnn[3] = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        cnn[4][1] = copy.deepcopy(cnn[5][0])
        cnn[4][1].conv1 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        cnn[4][1].bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        cnn[4][1].conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        cnn[4][1].bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        cnn[4][1].downsample[0] = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(2, 2), bias=False)
        cnn[4][1].downsample[1] = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        cnn[5][0].conv1.stride = (2, 1)
        cnn[5][0].downsample[0].stride = (2, 1)
        cnn[6][0].conv1.stride = (2, 1)
        cnn[6][0].downsample[0].stride = (2, 1)
        cnn[7][0].conv1.stride = (2, 1)
        cnn[7][0].downsample[0].stride = (2, 1)
        if d_bug == 'avgpool':
            cnn.add_module('avgPooling', nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
            print('==============> avgPooling')
        elif d_bug == 'maxpool':
            cnn.add_module('avgPooling', nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0))
            print('==============> maxpool')
        else:
            cnn.add_module('last_conv', nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=0, bias=False))
            cnn.add_module('last_bn', nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            cnn.add_module('last_relu', nn.ReLU(inplace=True))
            print('==============> conv')

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )



    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output


###########################################################################################################
class CRNN_res50_pp(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug='maxpool', rudc=True):
        print('CRNN_res50_pp')
        super(CRNN_res50_pp, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        net = torchvision.models.resnet50(pretrained=False)
        l1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
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
        cnn[5][0].conv2.stride = (2, 1)
        cnn[5][0].downsample[0].stride = (2, 1)
        cnn[6][0].conv2.stride = (2, 1)
        cnn[6][0].downsample[0].stride = (2, 1)
        cnn[7][0].conv2.stride = (2, 1)
        cnn[7][0].downsample[0].stride = (2, 1)
        cnn.add_module('maxPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(2048, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output


###########################################################################################################
class CRNN_res50_1(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False, d_bug='maxpool', rudc=True):
        print('CRNN_res50_1')
        super(CRNN_res50_1, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

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

        cnn = nn.Sequential(l1, *(list(net.children())[4: -2]))
        # 修改网络层
        cnn[1][0].conv2.stride = (2, 2)
        cnn[1][0].downsample[0].stride = (2, 2)
        cnn[2][0].conv2.stride = (2, 1)
        cnn[2][0].downsample[0].stride = (2, 1)
        cnn[3][0].conv2.stride = (2, 1)
        cnn[3][0].downsample[0].stride = (2, 1)
        cnn[4][0].conv2.stride = (2, 1)
        cnn[4][0].downsample[0].stride = (2, 1)
        cnn.add_module('maxPooling', nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(2048, nh, nh),
            BidirectionalLSTM(nh, nh, nclass)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        # add log_softmax to converge output
        # output = F.log_softmax(output, dim=2)

        return output