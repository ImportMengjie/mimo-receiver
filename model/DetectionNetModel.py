import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import math

from model import Tee
from model import BaseNetModel


class Lcg(nn.Module):

    def __init__(self, n_t: int, vector):
        super().__init__()
        if vector:
            self.alpha = nn.Parameter(torch.Tensor(2 * n_t, 1))
            self.beta = nn.Parameter(torch.Tensor(2 * n_t, 1))
        else:
            self.alpha = nn.Parameter(torch.Tensor(1))
            self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        self.alpha.data.zero_()
        self.beta.data.zero_()

    def forward(self, s, r, d, A):
        s_next = s + self.alpha * d
        r_next = r - self.alpha * (A @ d)
        d_next = r_next + self.beta * d
        return s_next, r_next, d_next


class DetectionNetModel(BaseNetModel):

    def __init__(self, n_r, n_t, layer_nums: int, vector=True, modulation='qpsk', is_training=True):
        super().__init__()
        self.n_r = n_r
        self.n_t = n_t
        self.vector = vector
        self.modulation = modulation
        self.layer_nums = layer_nums
        self.training_layer = layer_nums
        self.is_training = is_training
        self.lcg_layers = [Lcg(n_t, vector) for _ in range(self.layer_nums)]
        self.lcg_layers = nn.ModuleList(self.lcg_layers)
        self.fix_forward_layer = False

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

    def reset_requires_grad(self):
        self.fix_forward_layer = False
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, A, b):
        s = torch.zeros(b.shape)
        if torch.cuda.is_available():
            s = s.cuda()
        r = b
        d = r
        for i in range(self.training_layer):
            s, r, d = self.lcg_layers[i](s, r, d, A)
        return s,

    def __str__(self):
        return '{}_r{}t{}_v{}num{}m:{}'.format(self.__class__.__name__, self.n_r, self.n_t, self.vector,
                                               self.layer_nums, self.modulation)

    def get_train_state_str(self):
        return 'train layer:{}/{},fix forward:{}'.format(self.training_layer, self.layer_nums, self.fix_forward_layer)


class DetectionNetLoss(nn.Module):

    def __init__(self):
        super(DetectionNetLoss, self).__init__()
        self.i = 0

    def forward(self, x, x_hat):
        loss = F.mse_loss(x_hat, x)
        # if self.i % 1000 == 0:
        #     nmse = 10 * torch.log10((((x - x_hat) ** 2).sum(-1).sum(-1) / (x ** 2).sum(-1).sum(-1)).mean())
        #     print(nmse.item(), end='')
        # self.i += 1
        return loss


class DetectionNetTee(Tee):

    def __init__(self, items):
        super().__init__(items)
        self.A, self.b, self.x = items
        self.x_hat = None

    def get_model_input(self):
        return self.A, self.b

    def set_model_output(self, outputs):
        self.x_hat, = outputs

    def get_loss_input(self):
        return self.x, self.x_hat
