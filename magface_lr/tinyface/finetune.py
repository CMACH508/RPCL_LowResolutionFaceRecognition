# _date_:2021/11/4 14:58

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn
import argparse

torch.backends.cudnn.bencmark = True
from os.path import join

base_dir = os.path.abspath('../..')
add_path = [base_dir, join(base_dir, 'magface_lr')]
sys.path += add_path

from magface_lr.magface import SoftmaxBuilder, MagLoss
from magface_lr import data
from magface_lr.data import train_tinyface

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='1', help='')
parser.add_argument('--batch_size', type=int, default=80, help='')
parser.add_argument('--max_epoch', type=int, default=300, help='')
parser.add_argument('--base_lr', type=float, default=0.1, help='')
parser.add_argument('--lr_adjust', nargs='+', type=int, default=[10], help='')
parser.add_argument('--save_dir', type=str, default=None, help='')
parser.add_argument('--is_rpcl', type=int, default=1, help='')
parser.add_argument('--f', type=float, default=0.5, help='')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dir = os.path.abspath('../..')
if args.save_dir is None:
    args.save_dir = join(dir, 'rpcl-f05-t07/tinyface_tune_sota')
if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)


def train():
    device = torch.device("cuda:0")
    device_ids = list(range(torch.cuda.device_count()))
    # print(device_ids)
    num_class = 2570
    model = SoftmaxBuilder(class_num=num_class, is_rpcl=args.is_rpcl, f=args.f)
    # model = Resnet18(num_classes=6609)
    model = DataParallel(model, device_ids=device_ids).to(device)
    model.train()

    data.BASE_ROOT = os.path.abspath('../../dataset')

    # 优化器
    optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=args.base_lr, momentum=0.9, weight_decay=1e-5)  # 权值衰减: 加入L2正则?
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_adjust, gamma=0.1)

    criterion = MagLoss(is_rpcl=args.is_rpcl)

    tune_model_path = os.path.abspath('../../checkpoints/rpcl_magface_sota_f05t07.pth')
    ckpt = torch.load(tune_model_path)
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in ckpt['model'].items():
        model_v = model_state_dict[k]
        if v.shape == model_v.shape:
            new_state_dict[k] = v
        else:
            new_state_dict[k] = model_v
    model.load_state_dict(new_state_dict)
    print(tune_model_path)
    print('------------------ load model successfully -----------------')

    dataset = train_tinyface()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=5)

    batch_acc, loss_recorder, kl_loss_recorder = [], [], []
    acc_list = []

    for epo in range(args.max_epoch):
        tmp_acc = 0
        tmp_loss = 0
        acc_num = 0.0
        total_num = 0
        scheduler.step()
        total_loss = 0
        loss_num = 0

        for ite, (imgs, labels) in enumerate(dataloader):
            model.train()
            # print('img.shape:', img.shape)

            imgs = imgs.to(device)
            labels = labels.to(device)

            output, x_norm = model(imgs)

            loss_id, loss_g, one_hot = criterion(output, labels, x_norm)
            loss = loss_id + 20 * loss_g

            total_loss += loss.item()
            loss_num += imgs.shape[0]
            tmp_loss = total_loss / loss_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predy = np.argmax(output[0].data.cpu().numpy(), axis=1)  # TODO
            batch_acc = np.sum((predy == labels.data.cpu().numpy()).astype(int))
            total_num += imgs.shape[0]
            acc_num += batch_acc
            tmp_acc = acc_num / total_num

            if ite % 100 == 0:
                print('Epoch: {}/{}, ite:{}/{}, tmp_acc={:3f}, loss={:.3f} loss_id={:.3f} loss_g={:.3f} lr={:.4f}'.
                      format(epo, args.max_epoch, ite, len(dataloader), tmp_acc, tmp_loss, loss_id.item(), loss_g.item(), optimizer.param_groups[0]['lr']))

        tmp_acc = acc_num / total_num
        tmp_loss = total_loss / loss_num

        filename = 'epoch_{}_loss_{:.3f}_acc_{:.3f}.pth'.format(epo, tmp_loss, tmp_acc)
        savename = os.path.join(args.save_dir, filename)
        if epo > 60 or tmp_acc > 0.95:
            save_dict = {'model': model.state_dict()}
            torch.save(save_dict, savename)

        # acc_list.append(tmp_acc)
        # if len(acc_list) > 4:
        #     acc_list.pop(0)
        # if np.mean(acc_list) > 0.98:
        #     print(acc_list)
        #     break


if __name__ == '__main__':
    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    train()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
