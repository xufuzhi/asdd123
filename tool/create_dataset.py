import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob
from itertools import islice


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


# def writeCache(env, cache):
#     with env.begin(write=True) as txn:
#         for k, v in cache.iteritems():
#             txn.put(k, v)

# ### python3中修改为
def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(v) == str:
                v = v.encode()
            txn.put(k.encode(), v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert(len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    # map_size=1073741824 定义最大空间是1GB，如果在windows系统上报错改成map_size=8589934592
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i]
        label = labelList[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        ########## .mdb数据库文件保存了两种数据，一种是图片数据，一种是标签数据，它们各有其key
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        ########
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def read_text(path):
    with open(path) as f:
        text = f.read()
    text = text.strip()

    return text


if __name__ == '__main__':

    m = 'from name'     # 'from name'：标签从图片名字上获取。 'from txt': 标签从标签文件获取

    outputPath = '../data/lmdb_1w'     # lmdb 输出目录
    # 训练图片路径，标签是txt格式，名字跟图片名字要一致，如123.jpg对应标签需要是123.txt
    # path = '../data/*.jpg'
    path = '/home/xfz/Projects/PycharmProjects/TextRecognitionDataGenerator-master/trdg/out'    # 标签文件， 当m='from name'时候为图片文件夹位置

    imgPaths = []
    labellist = []
    if m == 'from name':
        # 获取该目录下所有文件，存入列表中
        f_nameList = os.listdir(path)
        for imgname in f_nameList:
            # 通过文件名后缀过滤文件类型
            exp = imgname.rsplit('.')[-1]
            if exp not in ['png', 'jpg']:
                continue
            label = imgname.split('_', 1)[0]
            imgPaths.append(os.path.join(path, imgname))
            labellist.append(label)
    elif m == 'from txt':
        rootpath = path.rsplit(os.path.sep, 1)[0]
        with open(path, encoding='utf-8-sig') as f:
            for line in f:
                imgname, label = line.split(',', 1)
                imgPaths.append(os.path.join(rootpath, 'images', imgname.strip()))
                labellist.append(label.strip().replace('"', ''))
    else:
        raise ValueError("m内容错误，支持的内容：'from name'， 'from txt'")




    # imagePathList = glob.glob(path)
    # print('一共%d张图片'%(len(imagePathList)))
    # imgLabelLists = []
    # for p in imagePathList:
    #     try:
    #         imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
    #     except:
    #         continue
    #
    # # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    # ##sort by lebelList
    # imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    # imgPaths = [p[0] for p in imgLabelList]
    # txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, labellist, lexiconList=None, checkValid=True)

    # ### 读取LMDB数据集中图片并显示出来，验证一下数据集是否制作成功
    val_num = 10
    with lmdb.open(outputPath) as env:
        txn = env.begin()
        for key, value in islice(txn.cursor(), val_num):
            imageBuf = np.fromstring(value, dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # 得到图片对应 label
                key = key.decode().replace('image', 'label', 1).encode()
                label = txn.get(key).decode()
                print(label)
                # 显示图片
                cv2.imshow('image', img)
                cv2.waitKey()
            else:   # 标签数据，不处理
                pass
                # print('key: %s    label: %s' % (key, value))
