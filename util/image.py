import torch
import opt
import torch.nn as nn
from opt import *


class UnMinMaxUnNormalization(nn.Module):
    def __init__(self, device):
        super(UnMinMaxUnNormalization, self).__init__()
        self.std = torch.Tensor(opt.STD).to(device)
        self.mean = torch.Tensor(opt.MEAN).to(device)
        self.max_value = torch.Tensor(opt.MAX).to(device)
        self.min_value = torch.Tensor(opt.MIN).to(device)

    def forward(self, x):
        x = x * self.std + self.mean
        x = x * self.max_value + self.min_value
        return x

# For visualization
max_transform = MAX[0]


def un_max_transform(x):
    return x * max_transform


def unnormalize(x):
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    # x = x * opt.STD + opt.MEAN
    x = un_max_transform(x)
    x = x.transpose(1, 3)
    return x


def unnormalize_vis(x):
    x = un_max_transform(x)
    x = x.transpose(1, 3)
    x = x * torch.Tensor(opt.STD) + torch.Tensor(opt.MEAN)
    x = x.transpose(1, 3)
    x /= (max_transform / 4)
    return x
