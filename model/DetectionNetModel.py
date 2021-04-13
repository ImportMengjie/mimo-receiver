import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import math

from model import Tee


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
        stdv = 1. / math.sqrt(self.alpha.shape[0])
        self.alpha.data.uniform_(-stdv, stdv)
        self.beta.data.uniform_(-stdv, stdv)

    def forward(self, s, r, d, A):
        s_next = s + self.alpha * d
        r_next = r - self.alpha * (A @ d)
        d_next = r_next + self.beta * d
        return s_next, r_next, d_next


class DetectionNetModel(nn.Module):

    def __init__(self, n_r, n_t, layer_nums: int, vector=True, modulation='qpsk'):
        super().__init__()
        self.n_r = n_r
        self.n_t = n_t
        self.vector = vector
        self.modulation = modulation
        self.layer_nums = layer_nums
        self.lcg_layer = Lcg(n_t, vector)

    def forward(self, A, b):
        s = torch.zeros(b.shape)
        r = b
        d = r
        for _ in range(self.layer_nums):
            s, r, d = self.lcg_layer(s, r, d, A)
        return s,

    def __str__(self):
        return '{}_r{}t{}_v{}num{}m:{}'.format(self.__class__.__name__, self.n_r, self.n_t, self.vector,
                                               self.layer_nums, self.modulation)


class DetectionNetLoss(nn.Module):

    def __init__(self):
        super(DetectionNetLoss, self).__init__()

    def forward(self, x, x_hat):
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
        self.x_hat, = outputs

    def get_loss_input(self):
        return self.x, self.x_hat
