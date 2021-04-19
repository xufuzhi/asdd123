import argparse
import torch
from torch.nn import CTCLoss
from tqdm import tqdm

import models.crnn as crnn
from utils import utils
from utils import dataset
import time


def val(net, dataset, criterion, labelConverter, batchSize=64, max_iter=0, n_display=500):
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
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batchSize, num_workers=0)
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

    accuracy = n_correct / float(total_img)
    print(f'Test loss: {loss_avg.val():.3f}, n_correct:{n_correct}, total_img: {total_img}, accuray: {accuracy:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batchSize', type=int, default=256, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
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
    dataset_val = dataset.Dataset_lmdb(root=opt.valroot, transform=dataset.ResizeNormalize((100, 32)))

    # 构建网络
    net_crnn = crnn.CRNN(opt.imgH, 1, len(alphabet) + 1, opt.nh).to(device=device)
    net_crnn.load_state_dict(torch.load(opt.weight))
    # print(net_crnn)

    str2label = utils.StrLabelConverter(alphabet)
    ctc_loss = CTCLoss(zero_infinity=True).to(device=device)

    # ### 开始验证
    val(net_crnn, dataset_val, ctc_loss, str2label, batchSize=256, max_iter=opt.max_iter)




