# pylint: disable=import-error,no-name-in-module

import torch.nn as nn
from torch.nn.init import xavier_normal, kaiming_normal


def sequential(*kargs):
    seq = nn.Sequential(*kargs)
    for layer in reversed(kargs):
        if hasattr(layer, "out_channels"):
            seq.out_channels = layer.out_channels
            break
        if hasattr(layer, "out_features"):
            seq.out_channels = layer.out_features
            break
    return seq


def weight_initialization(weight, init, activation):
    if init is None:
        return
    if init == "kaiming":
        assert not activation is None
        if hasattr(activation, "negative_slope"):
            kaiming_normal(weight, a=activation.negative_slope)
        else:
            kaiming_normal(weight, a=0)
    elif init == "xavier":
        xavier_normal(weight)
    return


def conv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    init="kaiming",
    activation=nn.ReLU(),
    use_batchnorm=False,
):
    convs = []
    if type(padding) == type(list()):
        assert len(padding) != 3
        if len(padding) == 4:
            convs.append(nn.ReflectionPad2d(padding))
            padding = 0

    # print(padding)
    convs.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    # weight init
    weight_initialization(convs[-1].weight, init, activation)
    # activation
    if not activation is None:
        convs.append(activation)
    # bn
    if use_batchnorm:
        convs.append(nn.BatchNorm2d(out_channels))
    seq = nn.Sequential(*convs)
    seq.out_channels = out_channels
    return seq


def deconv(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    output_padding=0,
    init="kaiming",
    activation=nn.ReLU(),
    use_batchnorm=False,
):
    convs = []
    convs.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding))
    # weight init
    weight_initialization(convs[0].weight, init, activation)
    # activation
    if not activation is None:
        convs.append(activation)
    # bn
    if use_batchnorm:
        convs.append(nn.BatchNorm2d(out_channels))
    seq = nn.Sequential(*convs)
    seq.out_channels = out_channels
    return seq


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
