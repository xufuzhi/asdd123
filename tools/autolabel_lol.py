import cv2 as cv
import os

import numpy as np

import models.crnn as crnn
import torch
from utils import utils
import torch.nn.functional as F


class Ocr():
    def __init__(self, weight):
        super(Ocr, self).__init__()
        # 读取字母表
        with open('../data/lol.alphabet', encoding='utf-8') as f:
            self.alphabet = f.read().strip()
        self.net = crnn.CRNN(32, 1, len(self.alphabet) + 1, 256).to('cuda')
        self.net.load_state_dict(torch.load(weight))
        self.converter = utils.StrLabelConverter(self.alphabet)


    def __call__(self, img):
        img = cv.resize(img,dsize=(100, 32))
        if img.shape[-1] == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = torch.from_numpy(img).to(torch.float32).unsqueeze(0).unsqueeze(-1).permute(0, 3, 1, 2).cuda() / 255

        # aa = img.cpu().squeeze().numpy()
        # cv.imshow('aa', aa), cv.waitKeyEx(), cv.destroyAllWindows()

        preds = self.net(img)
        preds = F.softmax(preds, dim=2)
        score, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([preds.size(0)])
        sim_pred = self.converter.decode(preds, preds_size, raw=False)

        score = score.squeeze().min().item()

        return sim_pred, score


def doimg(img):
    # mask = (img[..., 0] > 180) & (img[..., 1] > 180) & (img[..., 2] > 180)
    # img_ = np.where(np.tile(mask[..., np.newaxis], [1, 1, 3]), img, 50)
    # return img_

    mask = (img[..., 0] < 50) & (img[..., 1] > 100) & (img[..., 2] < 50)
    img_ = np.where(np.tile(mask[..., np.newaxis], [1, 1, 3]), 50, img)
    return img_





weight = '../weights/lol_3/netCRNN_ocr7_1c_best.pth'
# videopath = '/home/xfz/temps/LOLvideos/lol_768.mp4'
videopath = '/home/xfz/temps/LOLvideos/loltest_2.mp4'
outpath = '/home/xfz/temps/LOLdatasets/OCR/1'

# ### 1920*1080 分辨率下的位置
bboxes_1080 = [
            # [867, 1038, 906, 1053],   # 血条前半段
            # [913, 1038, 950, 1053],   # 血条后半段
            [867, 1036, 952, 1055],   # 血条总长
            # [1547, 1, 1622, 24],      # 队伍总击杀比分
            # [1663, 1, 1742, 24],      # 自己的击杀比分
            # [1854, 2, 1905, 23]       # 游戏时间
          ]
# ### 1366*768 分辨率下的位置
bboxes_768 = [
            [616, 737, 677, 751],   # 血条总长
            [1099, 1, 1152, 18],      # 队伍总击杀比分
            [1181, 1, 1238, 18],      # 自己的击杀比分
            [1317, 1, 1355, 18]       # 游戏时间
          ]
bboxes = bboxes_1080
imgname = 0

video = cv.VideoCapture(videopath)
ocr = Ocr(weight=weight)

# a = cv.imread('/home/xfz/Projects/PycharmProjects/crnn_pytorch/data/aaa/80.jpg')
# s = ocr(a)
# print(s, '    end')
# cv.imshow(f'{s}', a), cv.waitKeyEx(), cv.destroyAllWindows()


# assert not os.path.exists(os.path.join(outpath, 'right'))
# os.mkdir(os.path.join(outpath, 'right'))
# os.mkdir(os.path.join(outpath, 'wrong'))

# 清除labels.txt文件
with open(os.path.join(outpath, 'right_labels.txt'), 'w', encoding='utf-8') as f,\
        open(os.path.join(outpath, 'wrong_labels.txt'), 'w', encoding='utf-8') as wf:
    pass

word_all = []
for i in range(100000):
    flg, frame = video.read()
    if not flg:
        break
    if i % 20 != 0:
        continue

    # ### 截取需要的区域
    imgs = []
    for i, onebox in enumerate(bboxes):
        x1, y1, x2, y2 = onebox
        img = frame[y1: y2, x1: x2, ...]

        if i == 0:
            img = doimg(img)

        pred_str, score = ocr(img)
        if (score < 0.9) or (pred_str in word_all):
            continue
        print('score: ', score)

        # rimg = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=3)
        # cv.imshow('rect', rimg)
        txtimg = np.tile(np.ones_like(img), (2, 2, 1)) * 150
        x, y = 19, txtimg.shape[0] - 5
        if len(pred_str) > 8:
            x = 4
        cv.putText(txtimg, pred_str, (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.9, color=(0, 0, 0))
        txtimg = cv.resize(txtimg, (img.shape[1], img.shape[0]))
        img_show = np.concatenate((img, txtimg))
        img_show = cv.resize(img_show, (350, 150))
        cv.imshow('{pred_str}', img_show)
        # while True:
        #     k = cv.waitKeyEx()
        #     if k in [ord(' '), 65363]:
        #         break
        k = cv.waitKeyEx()
        print(k)
        if k == ord(' '):
            imname = os.path.join(outpath, 'right', str(imgname)) + '.jpg'
            cv.imwrite(imname, img)
            fline = imname.rsplit(os.sep, 1)[-1] + ', ' + pred_str + '\n'
            with open(os.path.join(outpath, 'right_labels.txt'), 'a', encoding='utf-8') as f:
                f.writelines(fline)
            print('add a right item: ', fline[:-1])
            word_all.append(pred_str)
        elif k == 65506:
            imname = os.path.join(outpath, 'wrong', str(imgname)) + '.jpg'
            cv.imwrite(imname, img)
            fline = imname.rsplit(os.sep, 1)[-1] + ',' + '\n'
            with open(os.path.join(outpath, 'wrong_labels.txt'), 'a', encoding='utf-8') as f:
                f.writelines(fline)
            print('add a wrong item: ', fline[:-1])
        else:
            continue

        imgname += 1


cv.destroyAllWindows()
# cv.imshow('1111111111111111111111', img)
# if ord(' ') != cv.waitKeyEx():
#     break

