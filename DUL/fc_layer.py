#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

device = torch.device('cuda')


class FullyConnectedLayer(nn.Module):

    def __init__(self, fc_mode='cosface', rival_margin=0.2):

        super(FullyConnectedLayer, self).__init__()

        self.fc_mode = fc_mode
        self.margin = 0.2
        self.rival_margin = rival_margin
        self.classnum = 10575
        self.in_feats = 512
        self.easy_margin = True
        self.scale = 32
        self.hard_mode = 'adaptive'
        self.t = 0.2

        self.weight = nn.Parameter(torch.Tensor(self.classnum, self.in_feats))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.cos_rival_m = math.cos(self.rival_margin)
        self.sin_rival_m = math.sin(self.rival_margin)

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

    def forward(self, x, label):

        cos_theta = F.linear(F.normalize(x), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        batch_size = label.size(0)
        cosin_simi = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)

        gpu_soft = F.softmax(cos_theta, dim=1)
        soft = gpu_soft.data

        max_index = soft.argmax(dim=1, keepdim=True)
        max_num = torch.gather(soft, 1, max_index)

        subtraction = torch.zeros(soft.shape).to(torch.device('cuda')).data

        new = torch.where(soft < max_num - 0.0001, soft, subtraction).data

        second_index = new.argmax(dim=1, keepdim=True).data

        with torch.no_grad():
            rival_index = torch.where(max_index == label.unsqueeze(1), second_index, max_index).data
            # print(factor_index.shape)
            rival_index = torch.squeeze(rival_index)
            rival_cosin_simi = cos_theta[torch.arange(0, batch_size), rival_index].view(-1, 1)

        if self.fc_mode == 'softmax':
            score = cosin_simi
            print('--', cosin_simi.shape, rival_cosin_simi.shape, label.shape, rival_index.shape)
            return cosin_simi, rival_cosin_simi

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
                # rival_score = torch.where(rival_cosin_simi > 0, rival_cosin_simi + self.margin, rival_cosin_simi)
                rival_score = torch.where(rival_cosin_simi > 0.6, (rival_cosin_simi + self.margin) * 0.2, rival_cosin_simi)

                # print(cosin_simi.shape, score.shape)
                # print(rival_cosin_simi.shape, rival_score.shape)
            else:
                score = cosin_simi - self.margin
                rival_score = rival_cosin_simi + self.margin

        # ''' ArcFace '''
        elif self.fc_mode == 'arcface':
            sin_theta = torch.sqrt(1.0 - torch.pow(cosin_simi, 2))
            cos_theta_m = cosin_simi * self.cos_m - sin_theta * self.sin_m

            rival_sin_theta = torch.sqrt(1.0 - torch.pow(rival_cosin_simi, 2))
            # rival_cos_theta_m = rival_cosin_simi * self.cos_rival_m + rival_sin_theta * self.sin_rival_m
            rival_cos_theta_m = rival_cosin_simi * self.cos_m + rival_sin_theta * self.sin_m
            rival_cos_theta_m = rival_cos_theta_m * 0.2

            if self.easy_margin:
                score = torch.where(cosin_simi > 0, cos_theta_m, cosin_simi)
                rival_score = torch.where(rival_cosin_simi > 0, rival_cos_theta_m, rival_cosin_simi)

                # print(cosin_simi.shape, score.shape)
                # print(rival_cosin_simi.shape, rival_score.shape)

            else:
                score = cos_theta_m
                rival_score = rival_cos_theta_m

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

        cos_theta.scatter_(1, rival_index.data.view(-1, 1), rival_score)

        # print('cos_rpcl:', cos_theta.grad_fn, rival_score.grad_fn)
        # print(cos_theta.grad, type(cos_theta))

        cos_theta.scatter_(1, label.data.view(-1, 1), score)

        # print('cos:', cos_theta.grad_fn, score.grad_fn)
        # print(cos_theta.grad, type(cos_theta))
        # print(label.grad_fn, factor_index.grad_fn, label.grad, factor_index.grad)

        cos_theta *= self.scale
        return cos_theta
