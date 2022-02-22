# pylint: disable=import-error,no-name-in-module

import sys
import torch
import torch.nn as nn
from .layer.residual import residual_unit_SE
from .layer import Flatten

sys.setrecursionlimit(1000000000)


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        # input
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU())
        # build blocks
        modules = []
        num_stages = 4
        filter_list = [64, 64, 128, 256, 512]
        units = [2, 2, 2, 2]
        inplanes = filter_list[0]
        for i in range(num_stages):
            modules.append(residual_unit_SE(inplanes, filter_list[i + 1], 2))
            inplanes = filter_list[i + 1] * modules[-1].expansion
            for j in range(units[i] - 1):
                modules.append(residual_unit_SE(inplanes, filter_list[i + 1], 1))

        self.body = nn.Sequential(*modules)

        # self.output_layer_mu = nn.Sequential(
        #     nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5),
        #     nn.Dropout(0.4),
        #     Flatten(),
        #     # nn.Linear(512 * modules[-1].expansion * 7 * 7, 512),
        #     nn.Linear(512 * modules[-1].expansion * 8 * 8, 512),
        #     nn.BatchNorm1d(512, eps=2e-5),
        # )

        self.flat = nn.Sequential(
            nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5),
            nn.Dropout(0.4),
            Flatten()
        )
        self.linear512 = nn.Linear(512 * modules[-1].expansion * 8 * 8, 512)
        self.bn512 = nn.BatchNorm1d(512, eps=2e-5)
        self.relu512 = nn.ReLU()

        self.linear2 = nn.Linear(512, 2)
        self.bn2 = nn.BatchNorm1d(2, eps=2e-5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, flag=None):
        x = self.input_layer(x)
        x = self.body(x)

        x = self.flat(x)
        x = self.linear512(x)
        x = self.bn512(x)
        # x = self.relu512(x)

        x = self.linear2(x)
        x = self.bn2(x)

        return x


class Resnet18(nn.Module):
    def __init__(self, num_classes=None, embedding_size=512):
        super(Resnet18, self).__init__()
        self.backbone = Backbone()
        print('----------- 512 2D -----------')

    def forward(self, x, flag=None):
        mu = self.backbone(x, flag=flag)
        return mu

        # if self.num_classes > 0:
        #     logits = self.classifier(embedding)
        # return mu, logvar, embedding, logits, last_conv
