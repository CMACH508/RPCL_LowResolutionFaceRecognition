#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn

torch.backends.cudnn.bencmark = True
from os.path import join

base_dir = os.path.abspath('..')
add_path = [base_dir]
sys.path += add_path

from RPCL.config import Config
from RPCL import eval as EVAL
from RPCL import faceloss
from RPCL import fc_layer_RPCL, fc_layer
from data import Dataset_CASIA
from models.resnet18 import Resnet18
from models.resnet34 import Resnet34
from models.resnet50 import Resnet50
from models.resnet64 import Resnet64


config = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)


def train():
    device = torch.device('cuda')
    print('backbone=', config.backbone)
    if config.backbone == 'resnet18':
        model = Resnet18(num_classes=10575, embedding_size=512)
    elif config.backbone == 'resnet34':
        model = Resnet34(num_classes=10575, embedding_size=512)
    elif config.backbone == 'resnet50':
        model = Resnet50(num_classes=10575, embedding_size=512)
    elif config.backbone == 'resnet64':
        model = Resnet64(num_classes=10575, embedding_size=512)

    model.to(device)
    model = DataParallel(model)

    if config.is_rival:
        metric = fc_layer_RPCL.FullyConnectedLayer(fc_mode=config.fc_mode)
    else:
        metric = fc_layer.FullyConnectedLayer(fc_mode=config.fc_mode)
    metric.to(device)
    metric = DataParallel(metric)

    face_loss = faceloss.FaceLoss()
    face_loss.to(device)
    # face_loss = DataParallel(face_loss)
    # print('model_ckpt_len=', len(model.state_dict()))

    resume = False
    if resume:
        ckpt = torch.load('checkpoint_path')
        model.load_state_dict(ckpt['backbone'])
        metric.load_state_dict(ckpt['metric'])
        print('------------------ load model successfully -----------------')

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}], lr=config.base_lr, weight_decay=config.weight_decay,
                                momentum=0.9, nesterov=True)
    # ''' only model '''
    # optimizer = torch.optim.SGD([{'params': model.parameters()}], lr=config.base_lr, weight_decay=config.weight_decay, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_adjust, gamma=0.1)

    dataset = Dataset_CASIA(config.train_root, config.train_list)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    batch_acc, loss_recorder, kl_loss_recorder = [], [], []
    max_lfw_acc, lfw_th, lfw_max_epo = 0, 0, 0
    # max_calfw_acc, calfw_th, calfw_max_epo = 0, 0, 0

    print('-------------- init')
    lfw_acc, lfw_std, lfw_thresh = EVAL.eval(model, test_root=config.test_root, test_list=config.test_list, test_batch_size=config.test_batch_size)

    for epo in range(config.max_epoch):
        scheduler.step()
        model.train()
        for ite, (img, label) in enumerate(dataloader, start=1):
            img = img.to(device)
            label = label.to(device)

            # print(label.shape)  # [300]
            # cls_loss = cls_criterion(logits, targets)
            # kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp_heatmap())) / mu.size()[0]
            # tot_loss = cls_loss + 0.01 * kl_loss
            # loss += tot_loss
            # cls_monitor += cls_loss
            # kl_monitor += kl_loss
            ''' no sigma '''
            mu, logvar, _, _, _ = model(img)
            mu = nn.functional.normalize(mu, dim=1)

            output = metric(mu, label)
            tot_loss = face_loss(output, label, one_hot_factor=None)

            ''' DUL + RPCL '''
            # mu, logvar, _, _, _ = model(img)
            # mu = nn.functional.normalize(mu, dim=1)
            # output = metric(mu, label)
            # cls_loss = face_loss(output, label, one_hot_factor=None)
            #
            # kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size()[0]
            # tot_loss = cls_loss + 0.001 * kl_loss

            optimizer.zero_grad()
            tot_loss.backward()
            tot_loss = tot_loss.item()
            optimizer.step()

            predy = np.argmax(output.data.cpu().numpy(), axis=1)  # TODO
            it_acc = np.mean((predy == label.data.cpu().numpy()).astype(int))
            if len(batch_acc) == 10:
                batch_acc.pop(0)
            batch_acc.append(it_acc)
            if len(loss_recorder) == 10:
                loss_recorder.pop(0)
            loss_recorder.append(tot_loss)

            if ite % config.print_freq == 0:
                print('epoch : %2d|%2d, iter:%4d|%4d,loss:%.4f,batch_ave_acc:%.4f,lr={%.4f}' %
                      (epo, config.max_epoch, ite, len(dataloader), np.mean(loss_recorder), np.mean(batch_acc),
                       optimizer.param_groups[0]['lr']))

                lfw_acc, lfw_std, lfw_thresh = EVAL.eval(model, test_root=config.test_root, test_list=config.test_list, test_batch_size=config.test_batch_size)
                if max_lfw_acc < lfw_acc:
                    print('%snew lfw SOTA was found%s' % ('*' * 16, '*' * 16))
                    print('epo={}, lfw_max_acc={}'.format(epo, lfw_acc))
                    max_lfw_acc = lfw_acc
                    lfw_th = lfw_thresh
                    lfw_max_epo = epo
                    filename = os.path.join(config.save_dir, 'lfw_sota.pth')
                    torch.save({
                        'epoch': epo,
                        'backbone': model.state_dict(),
                        'metric': metric.state_dict(),
                        'lfw_acc': lfw_acc,
                        'lfw_th': lfw_thresh
                    }, filename)

        if epo % 1 == 0:
            lfw_acc, lfw_std, lfw_thresh = EVAL.eval(model, test_root=config.test_root, test_list=config.test_list, test_batch_size=config.test_batch_size)
            filename = 'epoch_%d_lfw_%.4f.pth' % (epo, lfw_acc)
            savename = os.path.join(config.save_dir, filename)
            torch.save({
                'epoch': epo,
                'backbone': model.state_dict(),
                'metric': metric.state_dict(),
                'lfw_acc': lfw_acc,
                'lfw_th': lfw_thresh
            }, savename)
    print('the max lfw acc={}, lfw_th={}, epo={}'.format(max_lfw_acc, lfw_th, lfw_max_epo))


if __name__ == '__main__':
    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    train()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
