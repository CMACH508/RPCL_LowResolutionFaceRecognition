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
        self.output_layer_mu = nn.Sequential(
            # nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5, affine=False),
            nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5),
            nn.Dropout(0.4),
            Flatten(),
            # nn.Linear(512 * modules[-1].expansion * 7 * 7, 512),
            nn.Linear(512 * modules[-1].expansion * 8 * 8, 512),
            nn.BatchNorm1d(512, eps=2e-5)
        )
        self.output_layer_logvar = nn.Sequential(
            # nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5, affine=False),
            nn.BatchNorm2d(512 * modules[-1].expansion, eps=2e-5),
            nn.Dropout(0.4),
            Flatten(),
            # nn.Linear(512 * modules[-1].expansion * 7 * 7, 512),
            nn.Linear(512 * modules[-1].expansion * 8 * 8, 512),
            nn.BatchNorm1d(512, eps=2e-5),
        )

        # self.bn = nn.BatchNorm1d(512, eps=2e-5)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, flag=None):
        # x must be "RGB" format
        # x = (x - 127.5) * 0.0078125
        _ = "_"
        # print('flag==', flag)
        if flag == 'new_conv':
            mu = self.output_layer_mu(x)
            logvar = self.output_layer_logvar(x)
            embedding = self.reparameterize(mu, logvar)
            last_conv = x
            return mu, logvar, nn.functional.normalize(embedding, dim=1), last_conv

        x = self.input_layer(x)
        x = self.body(x)
        last_conv = x

        if flag == 'get_conv':
            return x, _, _, _

        mu = self.output_layer_mu(x)
        logvar = self.output_layer_logvar(x)
        embedding = self.reparameterize(mu, logvar)

        # print('before mu:', mu[0][:5].data.cpu().numpy())
        # mu = self.bn(mu)
        # print('after mu:', mu[0][:5].data.cpu().numpy())

        return mu, logvar, nn.functional.normalize(embedding, dim=1), last_conv


class Classifier(nn.Module):
    def __init__(self, num_classes, embedding_size):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, embeddings):
        kernel_norm = nn.functional.normalize(self.kernel, dim=0)
        logit = torch.mm(embeddings, kernel_norm)
        return logit


class Resnet18(nn.Module):
    def __init__(self, num_classes, embedding_size=512):
        super(Resnet18, self).__init__()
        # print('=================')
        self.backbone = Backbone()
        if num_classes > 0:
            self.classifier = Classifier(num_classes=num_classes, embedding_size=embedding_size)

        self.num_classes = num_classes

    def forward(self, x, flag=None):
        # print('model_input:', x.shape)
        mu, logvar, embedding, last_conv = self.backbone(x, flag=flag)
        logits = None
        return mu, logvar, embedding, logits, last_conv

        # if self.num_classes > 0:
        #     logits = self.classifier(embedding)
        # return mu, logvar, embedding, logits, last_conv
