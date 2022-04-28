import torch
import opt
import torch.nn as nn

max_transform = 25.540

def un_max_transform(x):
    return x * max_transform

class UnNormalization(nn.Module):
    def __init__(self, device):
        super(UnNormalization, self).__init__()
        self.std = torch.Tensor(opt.STD).to(device)
        self.mean = torch.Tensor(opt.MEAN).to(device)
        self.max_value = torch.Tensor(opt.MAX).to(device)
        self.min_value = torch.Tensor(opt.MIN).to(device)

    def forward(self, x):
        x = x * self.std + self.mean
        x = x * self.max_value + self.min_value
        return x


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
