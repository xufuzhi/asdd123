import argparse
import torch
from torch.nn import CTCLoss
from tqdm import tqdm

import models.crnn as crnn
from utils import utils
from utils import dataset
import time


def val(net, data_loader, criterion, labelConverter, batchSize=64, max_iter=0, n_display=10):
    """
    :param net:
    :param dataset:
    :param criterion: loss function
    :param labelConverter: label和文字转换器
    :param batchSize: 批次大小
    :param max_iter: 验证多少个iteration, max_iter=0时测试完整个数据集
    :param n_display: 打印多少条结果
    :return:
    """
    print('Start val...')
    device = next(net.parameters()).device  # get model device
    net.eval()

    val_iter = iter(data_loader)

    with torch.no_grad():
        image = torch.empty(0, dtype=torch.float32, device=device)
        text = torch.empty(0, dtype=torch.int32, device=device)
        length = torch.empty(0, dtype=torch.int32, device=device)
        n_correct = 0
        total_img = 0
        loss_avg = utils.Averager()
        max_iter = len(data_loader) if max_iter == 0 else min(max_iter, len(data_loader))
        for _ in tqdm(range(max_iter)):
            data = next(val_iter)
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            utils.loadData(image, cpu_images)
            t, l = labelConverter.encode(cpu_texts)
            utils.loadData(text, t)
            utils.loadData(length, l)
            preds = net(image)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            cost = criterion(preds, text, preds_size, length)
            loss_avg.add(cost)

            _, preds = preds.max(2)
            # preds = preds.squeeze(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = labelConverter.decode(preds.data, preds_size.data, raw=False)
            total_img += batch_size
            for pred, target in zip(sim_preds, cpu_texts):
                if pred == target.lower():
                    n_correct += 1

        raw_preds = labelConverter.decode(preds.data, preds_size.data, raw=True)[:n_display]
        for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
            print(f'{raw_pred:<{20}} => {pred:<{20}}, gt: {gt}')

    precision = n_correct / float(total_img)
    prt_msg = (f'Test loss: {loss_avg.val():.3f}  n_correct:{n_correct}  total_img: {total_img}  '
                f'precision: {precision:.3f}')
    vals = {'loss': loss_avg.val(), 'n_correct': n_correct, 'total_img': total_img, 'precision': precision}
    return vals, prt_msg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
    parser.add_argument('--imgC', type=int, default=3)
    parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
    parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
    # TODO(meijieru): epoch -> iter
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--weight', default='./weights/lmdb_5w/netCRNN_lastest.pth', help="path to weight file")
    parser.add_argument('--alphabet', type=str, default='./data/en.alphabet')
    parser.add_argument('--n_test_disp', type=int, default=50, help='Number of samples to display when test')
    parser.add_argument('--max_iter', type=int, default=0, help='最大iter次数，当为0时测试完整个数据集')
    opt = parser.parse_args()
    print(opt)

    if opt.cuda is None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')

    # 读取字母表
    with open(opt.alphabet, encoding='utf-8') as f:
        alphabet = f.read().strip()

    # ### 构建数据集对象
    dataset_val = dataset.Dataset_lmdb(root=opt.valroot, in_channels=opt.imgC)
    data_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True, batch_size=opt.batchSize,
                                              collate_fn=dataset.AlignCollate(imgH=opt.imgH, imgW=opt.imgW,
                                                                              keep_ratio=opt.keep_ratio),
                                              num_workers=0)
    # 构建网络
    net_crnn = crnn.CRNN(opt.imgH, opt.imgC, len(alphabet) + 1, opt.nh).to(device=device)
    net_crnn.load_state_dict(torch.load(opt.weight))
    # print(net_crnn)

    str2label = utils.StrLabelConverter(alphabet)
    ctc_loss = CTCLoss(zero_infinity=True).to(device=device)

    # ### 开始验证
    print(val(net_crnn, data_loader, ctc_loss, str2label, batchSize=opt.batchSize, max_iter=opt.max_iter, n_display=opt.n_test_disp))



