#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from IPython import embed


class FullyConnectedLayer(nn.Module):

    def __init__(self, fc_mode='cosface'):

        super(FullyConnectedLayer, self).__init__()

        self.fc_mode = fc_mode
        self.classnum = 10575
        self.in_feats = 512
        self.easy_margin = True
        self.scale = 32
        self.hard_mode = 'adaptive'
        self.t = 0.2

        if fc_mode == 'cosface':
            self.margin = 0.2
        elif fc_mode == 'arcface':
            self.margin = 0.5
        elif fc_mode == 'sphereface':
            self.margin = 2
        
        self.weight = nn.Parameter(torch.Tensor(self.classnum, self.in_feats))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin
        self.register_buffer('factor_t', torch.zeros(1))
        self.iter = 0
        self.base = 1000
        self.alpha = 0.0001
        self.power = 2
        self.lambda_min = 5.0
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

        if fc_mode == 'sphereface':
            self.margin = 2
        elif fc_mode == 'cosface':
            self.margin = 0.2

    def forward(self, x, label):

        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        batch_size = label.size(0)
        cosin_simi = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)

        if self.fc_mode == 'softmax':
            score = cosin_simi

        elif self.fc_mode == 'sphereface':
            self.iter += 1
            self.lamb = max(self.lambda_min, self.base * (1 + self.alpha * self.iter) ** (-1 * self.power))
            cos_theta_m = self.mlambda[int(self.margin)](cosin_simi)
            theta = cosin_simi.data.acos()
            k = ((self.margin * theta) / math.pi).floor()
            phi_theta = ((-1.0) ** k) * cos_theta_m - 2 * k
            score = (self.lamb * cosin_simi + phi_theta) / (1 + self.lamb)

        elif self.fc_mode == 'cosface':
            if self.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.margin, cosin_simi)
            else:
                score = cosin_simi - self.margin

        elif self.fc_mode == 'arcface':
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            if self.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = cos_theta_m

        elif self.fc_mode == 'mvcos':
            mask = cos_theta > cosin_simi - self.margin
            hard_vector = cos_theta[mask]
            if self.hard_mode == 'adaptive':
                cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # Adaptive
            else:
                cos_theta[mask] = hard_vector + self.t  # Fixed
            if self.easy_margin:
                score = torch.where(cosin_simi > 0, cosin_simi - self.margin, cosin_simi)
            else:
                score = cosin_simi - self.margin

        elif self.fc_mode == 'mvarc':
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            if self.hard_mode == 'adaptive':
                cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # Adaptive
            else:
                cos_theta[mask] = hard_vector + self.t  # Fixed
            if self.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
            else:
                score = cos_theta_m

        elif self.fc_mode == 'curface':
            with torch.no_grad():
                origin_cos = cos_theta
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m
            mask = cos_theta > cos_theta_m
            score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi - self.mm)
            hard_sample = cos_theta[mask]
            with torch.no_grad():
                self.factor_t = cos_theta_m.mean() * 0.01 + 0.99 * self.factor_t
            cos_theta[mask] = hard_sample * (self.factor_t + hard_sample)
        else:
            raise Exception('unknown fc type!')

        cos_theta.scatter_(1, label.data.view(-1, 1), score)
        cos_theta *= self.scale
        return cos_theta
