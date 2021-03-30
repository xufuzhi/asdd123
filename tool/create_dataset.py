import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import glob


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

    outputPath = '../data/lmdb'     # lmdb 输出目录
    # 训练图片路径，标签是txt格式，名字跟图片名字要一致，如123.jpg对应标签需要是123.txt
    path = '../data/*.jpg'

    imagePathList = glob.glob(path)
    print('一共%d张图片'%(len(imagePathList)))
    imgLabelLists = []
    for p in imagePathList:
        try:
            imgLabelLists.append((p, read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    createDataset(outputPath, imgPaths, txtLists, lexiconList=None, checkValid=True)

    # ### 读取LMDB数据集中图片并显示出来，验证一下数据集是否制作成功
    with lmdb.open(outputPath) as env:
        txn = env.begin()
        for key, value in txn.cursor():
            print (key, value)
            imageBuf = np.fromstring(value, dtype=np.uint8)
            img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                cv2.imshow('image', img)
                cv2.waitKey()
            else:
                print('This is a label: {}'.format(value))