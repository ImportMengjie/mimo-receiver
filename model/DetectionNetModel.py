import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import math

from loader import CsiDataloader
from model import Tee
from model import BaseNetModel

import utils.config as config


class Lcg(nn.Module):

    def __init__(self, n_t: int, svm):
        super().__init__()
        self.svm = svm
        if svm == 'v':
            self.alpha = nn.Parameter(torch.Tensor(2 * n_t, 1))
            self.beta = nn.Parameter(torch.Tensor(2 * n_t, 1))
        elif svm == 'm':
            self.alpha = nn.Parameter(torch.Tensor(2 * n_t, 2 * n_t))
            self.beta = nn.Parameter(torch.Tensor(2 * n_t, 2 * n_t))
        elif svm == 's':
            self.alpha = nn.Parameter(torch.Tensor(1))
            self.beta = nn.Parameter(torch.Tensor(1))
        else:
            raise Exception('only s v m not {}'.format(svm))
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha.data.zero_()
        self.beta.data.zero_()

    def forward(self, s, r, d, A):
        if self.svm == 'm':
            s_next = s + self.alpha @ d
            r_next = r - self.alpha @ (A @ d)
            d_next = r_next + self.beta @ d
        else:
            s_next = s + self.alpha * d
            r_next = r - self.alpha * (A @ d)
            d_next = r_next + self.beta * d
        return s_next, r_next, d_next


class DetectionNetModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, layer_nums: int, svm='s', modulation='qpsk', is_training=True,
                 extra=''):
        super().__init__(csiDataloader)
        self.n_r = csiDataloader.n_r
        self.n_t = csiDataloader.n_t
        self.svm = svm
        self.modulation = modulation
        self.layer_nums = layer_nums
        self.training_layer = layer_nums
        self.is_training = is_training
        self.lcg_layers = [Lcg(csiDataloader.n_t, svm) for _ in range(self.layer_nums)]
        self.lcg_layers = nn.ModuleList(self.lcg_layers)
        self.fix_forward_layer = False

        self.extra = extra
        self.name = self.__str__()

    def get_train_state(self):
        return {
            'train_layer': self.training_layer,
            'fix_forward': self.fix_forward_layer
        }

    def set_training_layer(self, training_layer: int, fix_forward_layer=True):
        assert self.is_training
        assert training_layer <= self.layer_nums
        self.training_layer = training_layer
        for i in range(self.training_layer - 1):
            for p in self.lcg_layers[i].parameters():
                p.requires_grad = not fix_forward_layer
        # for i in range(self.training_layer, self.layer_nums):
        #     for p in self.lcg_layers[i].parameters():
        #         p.requires_grad = False
        self.fix_forward_layer = fix_forward_layer

    def set_test_layer(self, test_layer: int):
        self.training_layer = test_layer

    def reset_requires_grad(self):
        self.fix_forward_layer = False
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, A, b):
        s = torch.zeros(b.shape)
        if config.USE_GPU:
            s = s.cuda()
        r = b
        d = r
        S = None
        for i in range(self.training_layer):
            s, r, d = self.lcg_layers[i](s, r, d, A)
            if S is None:
                S = s
            else:
                S = torch.cat((S, s), dim=-1)
        return s, S

    def __str__(self):
        return '{}-{}_r{}t{}_{}num{}m{}{}'.format(self.get_dataset_name(), self.__class__.__name__, self.n_r, self.n_t,
                                                  self.svm, self.layer_nums, self.modulation, self.extra)

    def basename(self):
        return 'detection'

    def get_train_state_str(self):
        return 'train layer:{}/{},fix forward:{}'.format(self.training_layer, self.layer_nums, self.fix_forward_layer)


class DetectionNetLoss(nn.Module):

    def __init__(self, use_layer_total_mse=False):
        super(DetectionNetLoss, self).__init__()
        self.use_layer_total_mse = use_layer_total_mse

    def forward(self, x, x_hat, X_hat):
        if self.use_layer_total_mse:
            loss = F.mse_loss(X_hat, x.expand(*X_hat.size()))
        else:
            loss = F.mse_loss(x_hat, x)
        return loss


class DetectionNetTee(Tee):

    def __init__(self, items):
        super().__init__(items)
        self.A, self.b, self.x = items
        self.x_hat = None

    def get_model_input(self):
        return self.A, self.b

    def set_model_output(self, outputs):
        self.x_hat, self.X_hat = outputs

    def get_loss_input(self):
        return self.x, self.x_hat, self.X_hat
