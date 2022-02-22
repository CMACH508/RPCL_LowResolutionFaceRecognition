#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel

torch.backends.cudnn.bencmark = True
from os.path import join

base_dir = os.path.abspath('..')
add_path = [base_dir, join(base_dir, 'DUL'), join(base_dir, 'models')]
sys.path += add_path

from DUL.config import Config
from DUL import fc_layer, fc_layer_old
from DUL import eval as EVAL
from DUL import faceloss
from data import Dataset_CASIA
from models.resnet18 import Resnet18
from models.resnet34 import Resnet34
from models.resnet50 import Resnet50


config = Config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
if not os.path.exists(config.save_dir):
    os.mkdir(config.save_dir)


def train():
    device = torch.device('cuda')
    if config.backbone == 'resnet18':
        model = Resnet18(num_classes=10575, embedding_size=512)
    elif config.backbone == 'resnet34':
        model = Resnet34(num_classes=10575, embedding_size=512)
    elif config.backbone == 'resnet50':
        model = Resnet50(num_classes=10575, embedding_size=512)
    model.to(device)
    model = DataParallel(model)

    if config.is_rpcl:
        metric = fc_layer.FullyConnectedLayer(fc_mode=config.fc_mode, rival_margin=config.rival_margin)
    else:
        metric = fc_layer_old.FullyConnectedLayer(fc_mode=config.fc_mode)

    metric.to(device)
    metric = DataParallel(metric)

    resume = False
    if resume:
        ckpt = torch.load('checkpoint_path')
        model.load_state_dict(ckpt['backbone'])
        metric.load_state_dict(ckpt['metric'])
        print('------------------ load model successfully -----------------')

    face_loss = faceloss.FaceLoss()
    face_loss.to(device)
    # face_loss = DataParallel(face_loss)

    ckpt = model.state_dict()
    print('model_ckpt_len=', len(ckpt))

    optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric.parameters()}], lr=config.base_lr, weight_decay=config.weight_decay,
                                momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.lr_adjust, gamma=0.1)

    dataset = Dataset_CASIA(config.train_root, config.train_list)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)

    batch_acc, loss_recorder, kl_loss_recorder = [], [], []
    max_lfw_acc, th = 0, 0
    lfw_acc, lfw_thresh = EVAL.eval(model, test_root=config.test_root, test_list=config.test_list, batch_size=config.test_batch_size)

    # writer = SummaryWriter(config.save_dir)  # 参数为指定存储路径
    print('kl_weight=', config.kl_weight)
    for epo in range(config.max_epoch):
        scheduler.step()
        for ite, (img, label) in enumerate(dataloader):
            img = img.to(device)
            label = label.to(device)

            # mu, logvar, logits = model(inputs)
            # cls_loss = cls_criterion(logits, targets)
            # kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp_heatmap())) / mu.size()[0]
            # tot_loss = cls_loss + 0.01 * kl_loss
            # loss += tot_loss
            # cls_monitor += cls_loss
            # kl_monitor += kl_loss

            mu, logvar, embedding, logits, _ = model(img)
            output = metric(embedding, label)
            cls_loss = face_loss(output, label)
            # print('mu.shape=', mu.shape)
            # print('mu.size[0]=', mu.size()[0])
            kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / mu.size()[0]  # (0,1)
            # kl_loss = (-0.5 * torch.sum(1 + logvar - (-0.51) - (mu.pow(2) + logvar.exp()) / 0.6)) / mu.size()[0]  # (0, 0.6)
            # kl_loss = (-0.5 * torch.sum(1 + logvar - (-0.22) - (mu.pow(2) + logvar.exp()) / 0.8)) / mu.size()[0]  # (0, 0.8)
            tot_loss = cls_loss + config.kl_weight * kl_loss
            # tot_loss = cls_loss + 0.001 * kl_loss
            # tot_loss = cls_loss + 0.0001 * kl_loss
            # tot_loss = cls_loss + 0.1 * kl_loss

            # print('total_loss=', tot_loss)
            # print('kl_loss=', kl_loss)
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
            if len(kl_loss_recorder) == 10:
                kl_loss_recorder.pop(0)
            kl_loss_recorder.append(kl_loss.item())
            # if ite % 100 == 0:
            #     writer.add_scalar('loss', loss, ite+epo*len(dataloader))
            if ite % config.print_freq == 0:
                print('epoch : %2d|%2d, iter:%4d|%4d,loss:%.4f,cl_loss:%.4f,kl_loss:%.4f,batch_ave_acc:%.4f,lr={%.4f}' %
                      (epo, config.max_epoch, ite, len(dataloader), np.mean(loss_recorder), cls_loss.mean().item(), np.mean(kl_loss_recorder), np.mean(batch_acc),
                       optimizer.param_groups[0]['lr']))

                lfw_acc, lfw_thresh = EVAL.eval(model, test_root=config.test_root, test_list=config.test_list, batch_size=config.test_batch_size)
                print('epo={}, acc={}, th={}'.format(epo, lfw_acc, lfw_thresh))
                if max_lfw_acc < lfw_acc:
                    print('%snew SOTA was found%s' % ('*' * 16, '*' * 16))
                    print('epo={}, max_acc={}'.format(epo, lfw_acc))
                    max_lfw_acc = lfw_acc
                    th = lfw_thresh
                    filename = os.path.join(config.save_dir, 'lfw_sota.pth')
                    torch.save({
                        'epoch': epo,
                        'backbone': model.state_dict(),
                        'metric': metric.state_dict(),
                        'lfw_acc': lfw_acc,
                        'lfw_th': lfw_thresh
                    }, filename)

        if epo % 1 == 0:
            filename = 'epoch_%d_lfw_%.4f.pth' % (epo, lfw_acc)
            savename = os.path.join(config.save_dir, filename)
            torch.save({
                'epoch': epo,
                'backbone': model.state_dict(),
                'metric': metric.state_dict(),
                'lfw_acc': lfw_acc,
                'lfw_th': lfw_thresh
            }, savename)
    # writer.close()
    print('the max acc={}, th={}'.format(max_lfw_acc, th))


if __name__ == '__main__':
    print('start time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    train()
    print('end time:', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
