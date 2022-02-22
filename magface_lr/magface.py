#!/usr/bin/env python
import sys

sys.path.append("..")
from magface_master.models import iresnet
# from collections import OrderedDict
# from termcolor import cprint
from torch.nn import Parameter
import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
import numpy as np
import math
import torch
import torch.nn as nn
import os


def load_features(arch, embedding_size):
    if arch == 'iresnet18':
        features = iresnet.iresnet18(
            pretrained=True,
            num_classes=embedding_size)
    elif arch == 'iresnet34':
        features = iresnet.iresnet34(
            pretrained=True,
            num_classes=embedding_size)
    elif arch == 'iresnet50':
        features = iresnet.iresnet50(
            pretrained=True,
            num_classes=embedding_size)
    elif arch == 'iresnet100':
        features = iresnet.iresnet100(
            pretrained=True,
            num_classes=embedding_size)
    else:
        raise ValueError()
    return features


class SoftmaxBuilder(nn.Module):
    def __init__(self, class_num, is_rpcl, f=0.5, thred=0.7):
        super(SoftmaxBuilder, self).__init__()
        embedding_size = 512
        self.features = load_features('iresnet18', embedding_size)

        scale = 32
        self.thred = thred

        if is_rpcl:
            self.fc = MagLinear_rpcl(embedding_size, class_num, f, self.thred, scale=scale)
        else:
            self.fc = MagLinear(embedding_size, class_num, scale=scale)

        self.l_a = 10
        self.u_a = 110
        self.l_margin = 0.45
        self.u_margin = 0.8
        # self.l_margin = args.l_margin
        # self.u_margin = args.u_margin
        # self.l_a = args.l_a
        # self.u_a = args.u_a

    def _margin(self, x):
        """generate adaptive margin
        """
        margin = (self.u_margin - self.l_margin) / \
                 (self.u_a - self.l_a) * (x - self.l_a) + self.l_margin
        return margin

    def forward(self, x, is_test=False):
        x = self.features(x)
        if is_test:
            return x
        logits, x_norm = self.fc(x, self._margin, self.l_a, self.u_a)
        return logits, x_norm


class MagLinear(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, scale=64.0, easy_margin=True):
        super(MagLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin

    def forward(self, x, m, l_a, u_a):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # print(ada_margin[0][0])
        # print(ada_margin[2][0])
        # print(ada_margin[1][0])

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(
                cos_theta > threshold, cos_theta_m, cos_theta - mm)
        # multiply the scale in advance
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m], x_norm


class MagLinear_rpcl(torch.nn.Module):
    """
    Parallel fc for Mag loss
    """

    def __init__(self, in_features, out_features, f, thred, scale=64.0, easy_margin=True):
        super(MagLinear_rpcl, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.scale = scale
        self.easy_margin = easy_margin

        self.f = f
        self.thred = thred
        print('self.f=', self.f)

    def forward(self, x, m, l_a, u_a):
        """
        Here m is a function which generate adaptive margin
        """
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(l_a, u_a)
        ada_margin = m(x_norm)
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)

        # print(ada_margin[0][0])
        # print(ada_margin[2][0])
        # print(ada_margin[1][0])

        # norm the weight
        weight_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        cos_theta = cos_theta.clamp(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        rival_margin = torch.tensor(0.45).requires_grad_(False) * self.f
        # rival_margin = ada_margin
        # rival_margin = ada_margin * 0.1
        rival_cos_theta_m = cos_theta * torch.cos(rival_margin) + sin_theta * torch.sin(rival_margin)
        rival_cos_theta_m = rival_cos_theta_m.clamp(-1, 1)

        # for k in range(3):
        #     print('cos_theta, cos_theta_m, rival_cos_theta_m',
        #           cos_theta[k][5].data.cpu().numpy(), cos_theta_m[k][5].data.cpu().numpy(), rival_cos_theta_m[k][5].data.cpu().numpy())

        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
            rival_cos_theta_m = torch.where(cos_theta > self.thred, rival_cos_theta_m, cos_theta)

        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)

            rival_cos_theta_m = torch.where(cos_theta > threshold, rival_cos_theta_m, cos_theta - mm)

        # multiply the scale in advance
        rival_cos_theta_m = self.scale * rival_cos_theta_m
        cos_theta_m = self.scale * cos_theta_m
        cos_theta = self.scale * cos_theta

        return [cos_theta, cos_theta_m, rival_cos_theta_m], x_norm


class MagLoss(torch.nn.Module):
    """
    MagFace Loss.
    """

    def __init__(self, is_rpcl):
        super(MagLoss, self).__init__()
        l_a = 10
        u_a = 110
        l_margin = 0.45
        u_margin = 0.8
        scale = 64.0
        self.is_rpcl = is_rpcl

        self.l_a = l_a
        self.u_a = u_a
        self.scale = scale
        self.cut_off = np.cos(np.pi / 2 - l_margin)
        self.large_value = 1 << 10

    def calc_loss_G(self, x_norm):
        g = 1 / (self.u_a ** 2) * x_norm + 1 / (x_norm)
        return torch.mean(g)

    def forward(self, input, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        if self.is_rpcl:
            cos_theta, cos_theta_m, rival_cos_theta_m = input
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, target.view(-1, 1), 1.0)

            with torch.no_grad():
                gpu_soft = F.softmax(cos_theta, dim=1)
                max_index = gpu_soft.argmax(dim=1, keepdim=True)
                gpu_soft.scatter_(1, max_index.view(-1, 1), -1)
                second_index = gpu_soft.argmax(dim=1, keepdim=True)
                rival_index = torch.where(max_index == target.unsqueeze(1), second_index, max_index)
                # print('second_index={} rival_index={} target_index={}'.format(second_index.shape, rival_index.shape, target.shape))

            rival_one_hot = torch.zeros_like(cos_theta)
            rival_one_hot.scatter_(1, rival_index.view(-1, 1), 1.0)

            output = one_hot * cos_theta_m + rival_one_hot * rival_cos_theta_m + (1.0 - one_hot - rival_one_hot) * cos_theta
            # output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
            loss = F.cross_entropy(output, target, reduction='mean')
            return loss.mean(), loss_g, one_hot

        else:
            cos_theta, cos_theta_m = input
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, target.view(-1, 1), 1.0)
            output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
            loss = F.cross_entropy(output, target, reduction='mean')
            return loss.mean(), loss_g, one_hot
