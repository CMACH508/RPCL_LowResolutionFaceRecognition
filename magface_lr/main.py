# _date_:2021/10/27 11:31

import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn
import argparse
import math

torch.backends.cudnn.bencmark = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(101)

from os.path import join

base_dir = os.path.abspath('..')
add_path = [base_dir, join(base_dir, 'magface_lr')]
sys.path += add_path

from magface_lr.magface import SoftmaxBuilder, MagLoss
from magface_lr import data
from magface_lr.data import Dataset_CASIA, Dataset_LFW
from magface_lr import eval
import time, datetime

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='')
parser.add_argument('--batch_size', type=int, default=80, help='')
# parser.add_argument('--batch_size', type=int, default=8, help='')
parser.add_argument('--test_batch_size', type=int, default=100, help='')
parser.add_argument('--base_lr', type=float, default=0.01, help='')
# parser.add_argument('--base_lr', type=float, default=1e-3, help='')
# parser.add_argument('--weight_decay', type=float, default=5e-4, help='')
parser.add_argument('--lr_adjust', nargs='+', type=int, default=[60], help='')
parser.add_argument('--max_epoch', type=int, default=25, help='')
parser.add_argument('--print_freq', type=int, default=300, help='')
parser.add_argument('--embedding_size', type=int, default=512, help='')
parser.add_argument('--save_dir', type=str, default=None, help='')
parser.add_argument('--data_root', type=str, default=None, help='')
parser.add_argument('--arch', type=str, default='iresnet18', help='')
parser.add_argument('--f', type=float, default=0.2, help='')
parser.add_argument('--thred', type=float, default=0.7, help='')
parser.add_argument('--is_rpcl', type=int, default=1, help='')

args = parser.parse_args()
print(args)

print('[[[[ rival_cos_theta_m.clamp(-1, 1) * self.f ]]]]]')

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dir = os.path.abspath('.')
if args.save_dir is None:
    args.save_dir = join(dir + 'rpcl')
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


def train():
    device = torch.device("cuda:0")
    device_ids = list(range(torch.cuda.device_count()))
    # print(device_ids)
    model = SoftmaxBuilder(class_num=10575, is_rpcl=args.is_rpcl, thred=args.thred, f=args.f)
    # model = Resnet18(num_classes=6609)
    model = DataParallel(model, device_ids=device_ids).to(device)
    model.train()

    # optimizer
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.base_lr, momentum=0.9, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_adjust, gamma=0.1)

    dataset = Dataset_CASIA()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=6)

    testset = Dataset_LFW()
    test_loader = DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, drop_last=False, num_workers=5)
    is_center = False
    max_acc = 0
    criterion = MagLoss(is_rpcl=args.is_rpcl)

    for epo in range(args.max_epoch):
        total_loss = 0.0
        total_size = 0
        scheduler.step()

        for ite, (imgs, labels) in enumerate(dataloader):
            model.train()

            imgs, labels = imgs.to(device), labels.to(device)
            output, x_norm = model(imgs)

            loss_id, loss_g, one_hot = criterion(output, labels, x_norm)
            loss = loss_id + 20 * loss_g

            # print('loss={:.3f} loss_id={:.3f} loss_g={:.3f}'.format(loss.item(), loss_id.item(), loss_g.item()))

            total_loss += loss.item()
            total_size += imgs.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ite % 300 == 0:
                print('Epoch: {}/{}, ite:{}/{}, loss={:.3f} loss_id={:.3f} loss_g={:.3f}'.
                      format(epo, args.max_epoch, ite, len(dataloader), loss.item(), loss_id.item(), loss_g.item()))
                acc = eval.eval(model, test_loader, is_center=is_center)[0]
                if max_acc < acc:
                    max_acc = acc
                    print('----------- SOTA found -----------', max_acc)
                    save_dict = {'model': model.state_dict()}
                    torch.save(save_dict, join(args.save_dir, 'sota.pth'))

        if epo % 2 == 0:
            acc = eval.eval(model, test_loader, is_center=is_center)[0]
            save_dict = {'model': model.state_dict()}
            torch.save(save_dict, join(args.save_dir, '{}_{}.pth'.format(epo, acc)))
            if max_acc < acc:
                max_acc = acc
                print('----------- SOTA found -----------', max_acc)
                save_dict = {'model': model.state_dict()}
                torch.save(save_dict, join(args.save_dir, 'sota.pth'))

    torch.save({'model': model.state_dict()}, join(args.save_dir, 'final.pth'))
    print('----------------------')
    print('max_acc=', max_acc)


def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


if __name__ == '__main__':
    print('start time:', get_time())
    s_time = datetime.datetime.now()
    train()
    print('end time:', get_time())
    e_time = datetime.datetime.now()
    res = e_time - s_time
    print('train {}days {}h'.format(res.days, int(res.seconds / 3600)))
