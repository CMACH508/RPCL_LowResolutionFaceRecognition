# pylint: disable=import-error,no-name-in-module
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.PReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class residual_unit_SE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, act=nn.PReLU, use_IR=True, use_SE=True):
        super(residual_unit_SE, self).__init__()
        self.expansion = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_IR = use_IR
        self.act = act
        if stride == 1:
            self.shortcut_layer = lambda x: x
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (1, 1), stride, bias=False), nn.BatchNorm2d(out_channels, eps=2e-5)
            )

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels, eps=2e-5),
            nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=2e-5),
            self.act(out_channels),
            nn.Conv2d(out_channels, out_channels, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=2e-5),
        )
        if use_SE:
            self.SE = SEModule(out_channels, 16)
        else:
            self.SE = None

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        if self.SE is not None:
            res = self.SE(res)
        if self.use_IR:
            return res + shortcut
        else:
            return self.act()(res + shortcut)


class BottleneckUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=2, act=nn.ReLU, use_IR=False, use_SE=True):
        super(BottleneckUnit, self).__init__()
        # if in_channel == out_channel and stride == 1:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        self.use_IR = use_IR
        self.use_SE = use_SE
        self.act = act
        if stride == 1:
            self.shortcut_layer = lambda x: x
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion, eps=2e-5),
            )

        self.res_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=2e-5),
            self.act(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=2e-5),
            self.act(),
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion, eps=2e-5),
            SEModule(out_channels * self.expansion, 16),
        )
        if use_SE:
            self.SE = SEModule(out_channels * self.expansion, 16)
        else:
            self.SE = None

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        # return res + shortcut
        if self.SE is not None:
            res = self.SE(res)
        if self.use_IR:
            return res + shortcut
        else:
            return self.act()(res + shortcut)
