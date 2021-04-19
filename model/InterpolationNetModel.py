import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from model import Tee


class ConvReluBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, padding):
        super(ConvReluBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class InterpolationNetModel(nn.Module):

    def __init__(self, n_r, n_t, num_conv_block=18, channel_num=64, kernel_size=(3, 3)):
        super(InterpolationNetModel, self).__init__()
        self.n_r = n_r
        self.n_t = n_t
        self.num_conv_block = num_conv_block
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[0] - 1) // 2)

        self.first_conv = nn.Conv2d(2, channel_num, kernel_size=kernel_size, padding=padding)
        blocks = [ConvReluBlock(channel_num, channel_num, kernel_size, padding) for _ in range(num_conv_block)]
        self.blocks = nn.Sequential(*blocks)
        self.back_conv = nn.Conv2d(channel_num, 2, kernel_size, padding=padding)

    def forward(self, x):
        residual = x
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.back_conv(x)
        out = torch.add(x, residual)
        return out,

    def __str__(self) -> str:
        return '{}_r{}t{}_block{}channel{}'.format(self.__class__.__name__, self.n_r, self.n_t, self.num_conv_block,
                                                   self.channel_num)


class InterpolationNetLoss(nn.Module):

    def __init__(self):
        super(InterpolationNetLoss, self).__init__()

    def forward(self, H, H_hat):
        mse_loss = F.mse_loss(H_hat, H)
        return mse_loss


class InterpolationNetTee(Tee):

    def __init__(self, items):
        super(InterpolationNetTee, self).__init__(items)
        self.h, self.H = items
        self.h_hat = None

    def get_model_input(self):
        return self.h,

    def set_model_output(self, outputs):
        self.h_hat, = outputs

    def get_loss_input(self):
        return self.H, self.h_hat