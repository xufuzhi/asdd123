import torch
import torchvision.transforms.functional

import utils.utils as utils
from utils import dataset
from PIL import Image
import time
import cv2 as cv

import models.crnn as crnn


model_path = 'weights/lol_/netCRNN_CRNN_1c_lastest.pth'
# model_path = './data/crnn.pth'
img_path = './data/3.jpg'
# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz-,\'\\(/!.$#:) @&%?=[];+'

# model = crnn.CRNN(32, 1, 37, 256)
model = crnn.CRNN(32, 1, 59, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.StrLabelConverter(alphabet)

def transformer(x):
    x = cv.resize(x, dsize=(100, 32))
    x = torchvision.transforms.functional.to_tensor(x).unsqueeze(0)
    return x


image = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()

############################## debug
# image = image.repeat(1, 1, 1, 4)
# image = (image > 0.6) / 1.0
# aa = image[0, ...].cpu().permute(1, 2, 0).numpy()
# cv.imshow('img', aa), cv.waitKeyEx(), cv.destroyAllWindows()
# print('image_size: ', image.shape)
#####################

model.eval()
preds = model(image)

torch.cuda.synchronize()
t1 = time.time()
preds = model(image)
torch.cuda.synchronize()
print('time: ', time.time() - t1)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds, preds_size, raw=True)
sim_pred = converter.decode(preds, preds_size, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
