本仓库是从 [meijieru](https://github.com/meijieru/crnn.pytorch) fork过来，对原仓库进行了修改，增加了制作训练数据集脚本，修改代码使其适应最新版本pytorch等库，以及其他修改。

Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Run demo
--------
A demo program can be found in ``demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is converted from auther offered one by ``tool``.
Put the downloaded model file ``crnn.pth`` into directory ``data/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![Example Image](./data/demo.jpg)

Expected output:
    loading pretrained model from ./data/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb

Train a new model
-----------------
1. Construct dataset following [origin guide](https://github.com/bgshih/crnn#train-a-new-model). If you want to train with variable length images (keep the origin ratio for example), please modify the `tool/create_dataset.py` and sort the image according to the text length.
2. Execute ``python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda``. Explore ``train.py`` for details.
