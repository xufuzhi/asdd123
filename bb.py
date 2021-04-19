import lmdb
import numpy as np
import cv2 as cv
from itertools import islice


filepath = './data/lmdb_2w'
# filepath = '../../datas/aug240w'
# ### 读取LMDB数据集中图片并显示出来，验证一下数据集是否制作成功
val_num = 10
with lmdb.open(filepath) as env:
    txn = env.begin()
    for key, value in islice(txn.cursor(), val_num):
        imageBuf = np.fromstring(value, dtype=np.uint8)
        img = cv.imdecode(imageBuf, cv.IMREAD_GRAYSCALE)
        if img is not None:
            # 得到图片对应 label
            key = key.decode().replace('image', 'label', 1).encode()
            label = txn.get(key).decode()
            print(label)
            # 显示图片
            cv.imshow('image', img)
            cv.waitKey()
        else:  # 标签数据，不处理
            pass
            # print('key: %s    label: %s' % (key, value))