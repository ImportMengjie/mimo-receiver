import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from loader import CsiDataloader
from loader import DataType
from loader import DenoisingNetDataset
from model import Tee
from model import BaseNetModel

import utils.config as config


class Padding(nn.Module):

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size[1]

    def forward(self, x):
        zeros = torch.zeros(x.shape[:-1] + (self.kernel_size - 1,))
        if config.USE_GPU:
            zeros = zeros.cuda()
        return torch.cat((x[:, :, :, :x.shape[-1] // 2], zeros, x[:, :, :, x.shape[-1] // 2:]), -1)


class ConvBnReluBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, padding):
        super(ConvBnReluBlock, self).__init__()
        assert padding[1] == 0
        self.padding = Padding(kernel_size)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(self.padding(x))))


class ConvReluBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, kernel_size, padding):
        super().__init__()
        assert padding[1] == 0
        self.padding = Padding(kernel_size)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(self.padding(x)))


class NoiseLevelEstimationModel(nn.Module):

    def __init__(self, n_r, n_t, conv_num=5, channel_num=32, kernel_size=(3, 3)):
        super(NoiseLevelEstimationModel, self).__init__()
        self.conv_num = conv_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        self.n_r = n_r
        self.n_t = n_t
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.first_conv = ConvReluBlock(1, self.channel_num, kernel_size, padding)
        self.conv_bns = [ConvBnReluBlock(channel_num, channel_num, kernel_size, padding) for _ in range(conv_num)]
        self.conv_bns = nn.Sequential(*self.conv_bns)
        self.back_conv = nn.Conv2d(self.channel_num, 1, kernel_size, padding=padding)

        self.fc = nn.Sequential(
            nn.Linear(2 * n_r * n_t, 2000),
            nn.ReLU(inplace=True),
            nn.Linear(2000, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 50),
            nn.ReLU(inplace=True),
            nn.Linear(50, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_bns(x)
        x = self.back_conv(Padding(self.kernel_size)(x))
        sigma = self.fc(x.view(x.size(0), -1))
        return sigma


class NonBlindDenosingModel(nn.Module):

    def __init__(self, n_r, n_t, conv_num=5, channel_num=32, kernel_size=(3, 3)):
        super(NonBlindDenosingModel, self).__init__()
        self.n_r = n_r
        self.n_t = n_t
        self.conv_num = conv_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.first_conv = ConvReluBlock(2, self.channel_num, kernel_size, padding)
        self.conv_bns = [ConvBnReluBlock(channel_num, channel_num, kernel_size, padding) for _ in range(conv_num)]
        self.conv_bns = nn.Sequential(*self.conv_bns)
        self.back_conv = nn.Conv2d(self.channel_num, 1, kernel_size, padding=padding)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.conv_bns(x)
        out = self.back_conv(Padding(self.kernel_size)(x))
        return out


class DenoisingNetBaseModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, use_noise_level, only_return_noise_level):
        super().__init__(csiDataloader)
        self.use_noise_level = use_noise_level
        self.only_return_noise_level = only_return_noise_level

    def set_only_return_noise_level(self, only_return_noise_level):
        self.only_return_noise_level = only_return_noise_level


class CBDNetBaseModel(DenoisingNetBaseModel):

    def __init__(self, csiDataloader: CsiDataloader, noise_level_conv_num=3, denosing_conv_num=3, channel_num=32,
                 kernel_size=(3, 3), use_noise_level=True, only_return_noise_level=False):
        super(CBDNetBaseModel, self).__init__(csiDataloader, use_noise_level, only_return_noise_level)
        self.n_r = csiDataloader.n_r
        self.n_t = csiDataloader.n_t
        self.noise_level_conv_num = noise_level_conv_num
        self.denosing_conv_num = denosing_conv_num
        self.channel_num = channel_num
        self.kernel_size = kernel_size

        self.noise_level = NoiseLevelEstimationModel(self.n_r, self.n_t, noise_level_conv_num, channel_num, kernel_size)
        self.denosing = NonBlindDenosingModel(self.n_r, self.n_t, denosing_conv_num, channel_num, kernel_size)
        self.name = self.base_name()

    def forward(self, x, sigma):
        x = torch.cat((x[:, :, :, 0], x[:, :, :, 1]), -1).unsqueeze(1)
        if self.use_noise_level:
            sigma = self.noise_level(x)
        if self.only_return_noise_level:
            return None, sigma
        sigma_map = sigma.unsqueeze(-1).repeat(1, self.n_r, 2 * self.n_t).unsqueeze(1)
        concat_x = torch.cat([sigma_map, x], dim=1)
        noise = self.denosing(concat_x)
        h_hat = (x - noise).squeeze().unsqueeze(-1)
        h_hat = torch.cat((h_hat[:, :, :h_hat.shape[-2] // 2, :], h_hat[:, :, h_hat.shape[-2] // 2:, :]), -1)
        return h_hat, sigma.squeeze()

    def base_name(self):
        return '{}-{}_r{}t{}_sigma{}denosing{}channel{}'.format(self.get_dataset_name(), self.__class__.__name__,
                                                                self.n_r, self.n_t,
                                                                self.noise_level_conv_num, self.denosing_conv_num,
                                                                self.channel_num)

    def __str__(self) -> str:
        return self.name

    def basename(self):
        return 'denoising'


class DenoisingNetLoss(nn.Module):

    def __init__(self, a=0.3):
        super().__init__()
        self.a = a

    def forward(self, h, h_hat, sigma, sigma_hat):
        sigma = sigma.squeeze()
        sigma_hat = sigma_hat.squeeze()
        l2_h_loss = F.mse_loss(h_hat, h)
        asym_loss = torch.mean(
            torch.abs(self.a - F.relu(sigma - sigma_hat)) * torch.pow(sigma - sigma_hat, 2))
        # asym_loss = F.mse_loss(sigma_map, sigma_map_hat)
        # if not torch.isnan(sigma_hat[0]):
        #     asym_loss = torch.mean((h_hat-h)**2/sigma_hat)
        # else:
        #     asym_loss = 0
        loss = l2_h_loss + 0.5 * asym_loss
        return loss


class DenoisingNetTee(Tee):

    def __init__(self, items):
        super().__init__(items)
        self.h_ls, self.h, self.sigma = items
        self.sigma = self.sigma.reshape(-1, 1)
        self.h_hat, self.sigma_hat = None, None

    def get_model_input(self):
        return self.h_ls, self.sigma

    def set_model_output(self, outputs):
        self.h_hat, self.sigma_hat = outputs

    def get_loss_input(self):
        return self.h, self.h_hat, self.sigma, self.sigma_hat


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = DenoisingNetDataset(csiDataloader, DataType.train, [50, 51])
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = CBDNetBaseModel(csiDataloader)
    criterion = DenoisingNetLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    model.train()
    model.double()
    for items in dataloader:
        tee = DenoisingNetTee(items)
        tee.set_model_output(model(*tee.get_model_input()))
        loss = criterion(*tee.get_loss_input())
        print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
