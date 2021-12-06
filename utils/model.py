import torch
import torch.nn as nn
import torch.utils.data

from config import config


class Padding(nn.Module):

    def __init__(self, kernel_size, use_2dim=False):
        super().__init__()
        self.kernel_size = kernel_size[1]
        self.use_2dim = use_2dim

    def forward(self, x):
        if self.use_2dim:
            return x
        zeros = torch.zeros(x.shape[:-1] + (self.kernel_size - 1,))
        if config.USE_GPU:
            zeros = zeros.cuda()
        return torch.cat((x[:, :, :, :x.shape[-1] // 2], zeros, x[:, :, :, x.shape[-1] // 2:]), -1)


class ConvBnReluBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, padding, use_2dim=False):
        super(ConvBnReluBlock, self).__init__()
        assert (not use_2dim and padding[1] == 0) or (use_2dim and padding[1] != 0)
        self.padding = Padding(kernel_size, use_2dim)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.padding(x))))


class ConvReluBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, padding, use_2dim=False):
        super().__init__()
        assert (not use_2dim and padding[1] == 0) or (use_2dim and padding[1] != 0)
        self.padding = Padding(kernel_size, use_2dim)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(self.padding(x)))
