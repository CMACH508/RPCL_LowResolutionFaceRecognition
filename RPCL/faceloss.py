#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# from IPython import embed

gpu_device = torch.device('cuda')
class FaceLoss(nn.Module):
    ''' Classic loss function for face recognition '''

    def __init__(self):

        super(FaceLoss, self).__init__()

        self.hard_ratio = 0.9
        self.loss_power = 2
        self.loss_mode = 'ce'
        # self.loss_mode = 'focal_loss'

    def forward(self, predy, target, feat_mu=None, feat_var=None, HR_mu=None, HR_var=None, one_hot_factor=1.0):

        loss = None
        if self.loss_mode == 'focal_loss':
            logp = F.cross_entropy(predy, target, reduce=False)
            prob = torch.exp(-logp)
            loss = ((1 - prob) ** self.loss_power * logp).mean()
        elif self.loss_mode == 'hardmining':
            batchsize = predy.shape[0]
            logp = F.cross_entropy(predy, label, reduce=False)
            inv_index = torch.argsort(-logp)  # from big to small
            num_hard = int(self.hard_ratio * batch_size)
            hard_idx = ind_sorted[:num_hard]
            loss = torch.sum(F.cross_entropy(pred[hard_idx], label[hard_idx]))
        else:
            # loss = self.rpcl(input=predy, target=target, one_hot_factor=one_hot_factor)
            loss = F.cross_entropy(predy, target)

        if (HR_mu is not None) and (HR_var is not None):
            loss_mat = -(1 + torch.log(feat_var / HR_var) - (feat_var + (feat_mu - HR_mu) * (feat_mu - HR_mu)) / HR_var) / 2
            kl_loss = loss_mat.sum(dim=1).mean()
            # print('loss=', loss.mean().item(), 'kl_loss=', kl_loss.mean().item())
            loss = loss + 0.002 * kl_loss

        # else:
        # if (feat_mu is not None) and (feat_var is not None):
        #     loss_mat = -(1 + torch.log(feat_var) - feat_mu * feat_mu - feat_var) / 2
        #     kl_loss = loss_mat.sum(dim=1).mean()
        #     print('loss=', loss.mean().item(), 'kl_loss=', kl_loss.mean().item(), 'factor=', self.args.kl_tdoff)  # 0.01
        #     loss = loss + self.args.kl_tdoff * kl_loss

        return loss

    def rpcl(self, input, target, one_hot_factor=1.0):
        # print('*' * 20)
        # print(input)
        # print(target, target.shape)
        # loss = F.cross_entropy(input, target)
        # print('loss=', loss)
        soft = F.softmax(input, dim=1)
        # print('soft:', soft)

        max_index = soft.argmax(dim=1, keepdim=True).cuda()
        # print('max_index:', max_index.squeeze().tolist())
        # max_num = torch.gather(soft, 1, torch.LongTensor([[4],[3],[5]]))
        max_num = torch.gather(soft, 1, max_index)
        # print('max_num:', max_num)

        subtraction = torch.ones(soft.shape).cuda() * 0
        new = torch.where(soft < max_num - 0.0001, soft, subtraction).cuda()
        # print('new:', new)

        second_index = new.argmax(dim=1, keepdim=True).cuda()
        # print('second_index:', second_index.squeeze().tolist())

        # target = target.unsqueeze(1)
        # print('target:', target)

        factor_index = torch.where(max_index == target.unsqueeze(1), second_index, max_index).cuda()
        # print('factor_index:', factor_index)
        one_hot = torch.ones(soft.shape).cuda().scatter_(1, factor_index, one_hot_factor)
        # print('one_hot:', one_hot)

        # print('-------------------------')
        # print('old_input:', input[0][:10])
        input = input * one_hot
        # print('new_input:', input[0][:10], input.shape)

        # print('orig loss=', loss)
        loss = F.cross_entropy(input, target)
        # print('new loss:', loss)
        return loss