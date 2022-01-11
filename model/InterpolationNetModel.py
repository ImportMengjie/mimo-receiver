import torch.nn.functional as F
import torch.utils.data
import numpy as np

from loader import CsiDataloader
from model import Tee
from model import BaseNetModel
from utils import get_interpolation_pilot_idx
from utils import complex2real
from utils.model import *


class NoiseLevelModel(nn.Module):

    def __init__(self, n_sc, n_r, conv_num, channel_num, dnns, kernel_size, use_2dim):
        super().__init__()
        self.n_sc = n_sc
        self.n_r = n_r
        self.conv_num = conv_num
        self.channel_num = channel_num
        self.dnns = dnns
        self.kernel_size = kernel_size
        self.use_2dim = use_2dim
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2 if use_2dim else 0)
        self.first_conv = ConvReluBlock(2 if use_2dim else 1, channel_num, kernel_size, padding, use_2dim)
        self.conv_bn_relu_seq = [ConvBnReluBlock(channel_num, channel_num, kernel_size, padding, use_2dim) for _ in
                                 range(conv_num)]
        self.conv_bn_relu_seq = nn.Sequential(*self.conv_bn_relu_seq)
        self.back_conv = nn.Conv2d(self.channel_num, 1, kernel_size, padding=padding)

        if use_2dim:
            fc = [n_sc * n_r] + list(dnns) + [1, ]
        else:
            fc = [2 * n_sc * n_r, ] + list(dnns) + [1, ]
        self.fc = []
        for i, j in zip(fc[:-1], fc[1:]):
            self.fc.append(nn.Linear(i, j))
            self.fc.append(nn.ReLU(inplace=True))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_bn_relu_seq(x)
        x = self.back_conv(Padding(self.kernel_size, self.use_2dim)(x))
        sigma = self.fc(x.view(x.size(0), -1))
        return sigma


class NonBlindDenoisingModel(nn.Module):

    def __init__(self, conv_num, channel_num, kernel_size, add_var):
        super().__init__()
        self.conv_num = conv_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.add_var = add_var
        use_2dim = True
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2 if use_2dim else 0)
        self.first_conv = ConvReluBlock(3 if add_var else 2, channel_num, kernel_size, padding, use_2dim)
        self.conv_bn_relu_seq = [ConvBnReluBlock(channel_num, channel_num, kernel_size, padding, use_2dim) for _ in
                                 range(conv_num)]
        self.conv_bn_relu_seq = nn.Sequential(*self.conv_bn_relu_seq)
        self.back_conv = nn.Conv2d(self.channel_num, 2, kernel_size, padding=padding)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_bn_relu_seq(x)
        out = self.back_conv(Padding(self.kernel_size, True)(x))
        return out


class CBDNetSFModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, chuck_name, add_var, n_f=0, conv=6, channel=64, kernel_size=(3, 3),
                 extra=''):
        super().__init__(csiDataloader)
        self.conv = conv
        self.channel = channel
        self.kernel_size = kernel_size
        self.extra = extra
        self.denoising = NonBlindDenoisingModel(conv, channel, kernel_size, add_var)
        self.add_var = add_var
        self.chuck_name = chuck_name
        self.n_f = n_f
        self.name = self.__str__()

    def set_path(self, path):
        pass

    def __str__(self):
        name = '{}-{}_{}_r{}t{}K{}n_f{}_cv{}-ch{}-var{}-{}'.format(self.get_dataset_name(), self.chuck_name,
                                                                self.__class__.__name__, self.n_r, self.n_t, self.n_sc,
                                                                self.n_f, self.conv, self.channel, self.add_var,self.extra)
        return name

    def basename(self):
        return 'interpolation'

    def get_short_name(self):
        return 'CBD-SF'

    def forward(self, x, var):
        """
        :param x: batch, N_sc, N_r, 2
        :param var: batch,1
        :return:
        """
        x = x.permute(0, 3, 1, 2)
        var = var.reshape((-1, 1, 1, 1)).repeat(1, 1, self.n_sc, self.n_r)
        if self.add_var:
            concat_x = torch.cat([x, var], dim=1)
        else:
            concat_x = x
        noise = self.denoising(concat_x)
        g_hat = (x - noise)
        g_hat = g_hat.permute(0, 2, 3, 1)
        return g_hat,


class InterpolationNetModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, pilot_count: int = 3, num_conv_block=18, channel_num=64,
                 kernel_size=(3, 3), extra=''):
        super(InterpolationNetModel, self).__init__(csiDataloader)
        self.pilot_count = pilot_count
        self.num_conv_block = num_conv_block
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)

        self.first_conv = nn.Conv2d(2, channel_num, kernel_size=kernel_size, padding=padding)
        blocks = [ConvReluBlock(channel_num, channel_num, kernel_size, padding, use_2dim=True) for _ in
                  range(num_conv_block)]
        self.blocks = nn.Sequential(*blocks)
        self.back_conv = nn.Conv2d(channel_num, 2, kernel_size, padding=padding)

        self.extra = extra
        self.name = self.__str__()

    def forward(self, x):
        residual = x
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.back_conv(x)
        out = torch.add(x, residual)
        return out,

    def __str__(self) -> str:
        return '{}-{}_r{}t{}sc{}p{}_block{}channel{}{}'.format(self.get_dataset_name(), self.__class__.__name__,
                                                               self.n_r,
                                                               self.n_t, self.n_sc,
                                                               self.pilot_count, self.num_conv_block, self.channel_num,
                                                               self.extra)

    def basename(self):
        return 'interpolation'


class InterpolationNetLoss(nn.Module):

    def __init__(self,):
        super(InterpolationNetLoss, self).__init__()

    def forward(self, g, g_hat):
        loss = F.mse_loss(g_hat, g)
        return loss


class InterpolationNetTee(Tee):

    def __init__(self, items):
        super(InterpolationNetTee, self).__init__(items)
        self.h, self.H, self.var = items
        self.h_hat, self.var_hat = None, None

    def get_model_input(self):
        return self.h, self.var

    def set_model_output(self, outputs):
        self.h_hat, = outputs

    def get_loss_input(self):
        return self.H, self.h_hat,
