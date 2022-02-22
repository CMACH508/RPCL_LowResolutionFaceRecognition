#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


# from IPython import embed


class FaceLoss(nn.Module):
    ''' Classic loss function for face recognition '''

    def __init__(self):

        super(FaceLoss, self).__init__()

        self.hard_ratio = 0.9
        self.loss_power = 2
        self.loss_mode = 'ce'
        # self.loss_mode = 'focal_loss'

    def forward(self, predy, target, feat_mu=None, feat_var=None, HR_mu=None, HR_var=None):

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
