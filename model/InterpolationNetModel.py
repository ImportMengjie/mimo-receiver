import torch
import torch.nn as nn
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

    def __init__(self, conv_num, channel_num, kernel_size, use_2dim):
        super().__init__()
        self.conv_num = conv_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.use_2dim = use_2dim
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2 if use_2dim else 0)
        self.first_conv = ConvReluBlock(3 if use_2dim else 2, channel_num, kernel_size, padding, use_2dim)
        self.conv_bn_relu_seq = [ConvBnReluBlock(channel_num, channel_num, kernel_size, padding, use_2dim) for _ in
                                 range(conv_num)]
        self.conv_bn_relu_seq = nn.Sequential(*self.conv_bn_relu_seq)
        self.back_conv = nn.Conv2d(self.channel_num, 2 if use_2dim else 1, kernel_size, padding=padding)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_bn_relu_seq(x)
        out = self.back_conv(Padding(self.kernel_size, self.use_2dim)(x))
        return out


class CBDNetSFModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, pilot_count, noise_level_conv=4, noise_channel=32,
                 noise_dnn=(2000, 200, 50),
                 denoising_conv=6, denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=False,
                 only_return_noise_level=False, extra='', dft_chuck=0, use_dft_padding=False):
        super().__init__(csiDataloader)
        assert not (dft_chuck > 0 and use_dft_padding)

        self.pilot_idx = get_interpolation_pilot_idx(csiDataloader.n_sc, pilot_count)
        self.pilot_count = torch.sum(self.pilot_idx).item()
        self.noise_level_conv = noise_level_conv
        self.noise_channel = noise_channel
        self.noise_dnn = noise_dnn
        self.denoising_conv = denoising_conv
        self.denoising_channel = denoising_channel
        self.kernel_size = kernel_size
        self.use_two_dim = use_two_dim
        self.use_true_sigma = use_true_sigma
        self.only_return_noise_level = only_return_noise_level
        self.extra = extra
        self.noise_level = NoiseLevelModel(self.n_sc, self.n_r, noise_level_conv, noise_channel, noise_dnn, kernel_size,
                                           use_two_dim)
        self.denoising = NonBlindDenoisingModel(denoising_conv, denoising_channel, kernel_size, use_two_dim)
        self.dft_chuck = dft_chuck
        self.use_dft_padding = use_dft_padding
        if self.dft_chuck > 0:
            self.chuck_array = np.concatenate((np.ones(self.dft_chuck), np.zeros(self.n_sc - self.dft_chuck)))
            self.chuck_array = self.chuck_array.reshape((-1, 1))

        self.name = self.__str__()

    def __str__(self):
        name = '{}-{}_r{}t{}K{}p{}_cn{}-{}ch{}-{}dn{}k{}-{}_2dim{}_{}'.format(self.get_dataset_name(),
                                                                              self.__class__.__name__, self.n_r,
                                                                              self.n_t, self.n_sc, self.pilot_count,
                                                                              self.denoising_conv,
                                                                              self.noise_level_conv,
                                                                              self.denoising_channel,
                                                                              self.noise_channel,
                                                                              '-'.join(map(str, self.noise_dnn)),
                                                                              self.kernel_size[0], self.kernel_size[1],
                                                                              self.use_two_dim, self.extra)
        if self.dft_chuck > 0:
            name += '_chuck{}'.format(self.dft_chuck)
        if self.use_dft_padding:
            name += '_dft_padding'
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
        if self.use_dft_padding:
            x = x.cpu().numpy()
            x = x[:, self.pilot_idx]
            x = x[:, :, :, 0] + x[:, :, :, 1] * 1j
            x = np.fft.ifft(x, axis=-2)
            x = np.concatenate((x, np.zeros(x.shape[:1] + (self.n_sc - self.pilot_count, x.shape[-1]))), axis=-2)
            x = np.fft.fft(x, axis=-2)
            x = torch.from_numpy(x)
            x = complex2real(x)
            if config.USE_GPU:
                x = x.cuda()
        if self.dft_chuck > 0:
            x = x.cpu().numpy()
            x = x[:, :, :, 0] + x[:, :, :, 1] * 1j
            x = np.fft.ifft2(x)
            x = x * self.chuck_array
            x = np.fft.fft2(x)
            x = torch.from_numpy(x)
            x = complex2real(x)
            if config.USE_GPU:
                x = x.cuda()
        if not self.use_two_dim:
            x = torch.cat((x[:, :, :, 0], x[:, :, :, 1]), -1).unsqueeze(1)
            sigma = (var / 2) ** 0.5
        else:
            x = x.permute(0, 3, 1, 2)
            sigma = var ** 0.5
        if not self.use_true_sigma:
            sigma = self.noise_level(x)

        assert sigma.shape[1] == 1
        if self.only_return_noise_level:
            return None, sigma
        if self.use_two_dim:
            sigma_map = sigma.unsqueeze(-1).repeat(1, self.n_sc, self.n_r).unsqueeze(1)
        else:
            sigma_map = sigma.unsqueeze(-1).repeat(1, self.n_sc, 2 * self.n_r).unsqueeze(1)
        concat_x = torch.cat([x, sigma_map], dim=1)
        noise = self.denoising(concat_x)
        h_hat = (x - noise)
        if not self.use_two_dim:
            h_hat = h_hat.squeeze().unsqueeze(-1)
            h_hat = torch.cat((h_hat[:, :, :h_hat.shape[-2] // 2, :], h_hat[:, :, h_hat.shape[-2] // 2:, :]), -1)
        else:
            h_hat = h_hat.permute(0, 2, 3, 1)
        if self.use_two_dim:
            var_hat = sigma ** 2
        else:
            var_hat = 2 * (sigma ** 2)
        return h_hat, var_hat.squeeze()


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

    def __init__(self, only_train_noise_level, use2dim):
        super(InterpolationNetLoss, self).__init__()
        self.only_train_noise_level = only_train_noise_level
        self.use2dim = use2dim

    def forward(self, h, h_hat, var, var_hat):
        if self.use2dim:
            sigma = var ** 0.5
            sigma_hat = var_hat ** 0.5
        else:
            sigma = (var / 2) ** 0.5
            sigma_hat = (var_hat / 2) ** 0.5

        if self.only_train_noise_level:
            loss = ((sigma - sigma_hat) ** 2).mean()
        else:
            sigma = sigma.squeeze()
            sigma_hat = sigma_hat.squeeze()
            l2_sigma_loss = F.mse_loss(sigma, sigma_hat)
            l2_h_loss = F.mse_loss(h_hat, h)
            loss = l2_h_loss + l2_sigma_loss
        return loss


class InterpolationNetTee(Tee):

    def __init__(self, items):
        super(InterpolationNetTee, self).__init__(items)
        self.h, self.H, self.var = items
        self.h_hat, self.var_hat = None, None

    def get_model_input(self):
        return self.h, self.var

    def set_model_output(self, outputs):
        self.h_hat, self.var_hat = outputs

    def get_loss_input(self):
        return self.H, self.h_hat, self.var, self.var_hat
