import torch
import utils
import dataset
from PIL import Image
import time

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.jpg'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
# image = Variable(image)

model.eval()
preds = model(image)

torch.cuda.synchronize()
t1 = time.time()
preds = model(image)
torch.cuda.synchronize()
t2 = time.time() - t1
print('time: ', t2)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds, preds_size, raw=True)
sim_pred = converter.decode(preds, preds_size, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
