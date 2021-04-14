import lmdb
import numpy as np
import cv2 as cv


filepath = '../../datas/aug_vgg_synthtext240w'
# ### 读取LMDB数据集中图片并显示出来，验证一下数据集是否制作成功
with lmdb.open(filepath) as env:
    txn = env.begin()
    for key, value in txn.cursor():
        # print (key, value)
        imageBuf = np.fromstring(value, dtype=np.uint8)
        img = cv.imdecode(imageBuf, cv.IMREAD_GRAYSCALE)
        if img is not None:
            pass
            cv.imshow('image', img)
            cv.waitKey()
        else:
            print('key: %s    label: %s' % (key, value))