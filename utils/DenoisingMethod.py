import abc
from loader.CsiDataloader import CsiDataloader
import torch

from model import DenoisingNetModel
from utils import complex2real
from utils import conj_t

import utils.config as config


class DenoisingMethod(abc.ABC):

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_h_hat(self, y, h, x, var, rhh):
        pass

    def get_nmse(self, y, h, x, var, rhh):
        h_hat = self.get_h_hat(y, h, x, var, rhh)
        nmse = ((torch.abs(h - h_hat) ** 2).sum(-1).sum(-1) / (torch.abs(h) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()


class DenoisingMethodLS(DenoisingMethod):

    def get_key_name(self):
        return 'LS'

    def get_h_hat(self, y, h, x, var, rhh):
        h_ls = y @ torch.inverse(x)
        return h_ls


class DenoisingMethodMMSE(DenoisingMethod):

    def get_key_name(self):
        return 'NonIdeal-MMSE'

    def get_h_hat(self, y, h, x, var, rhh):
        n_r = y.shape[-2]
        n_t = x.shape[-2]
        I = torch.eye(n_t, n_t)
        h_hat = y @ torch.inverse(conj_t(x) @ rhh @ x + n_r * var * I) @ conj_t(
            x) @ rhh
        return h_hat


class DenoisingMethodIdealMMSE(DenoisingMethod):

    def get_key_name(self):
        return 'Ideal-MMSE'

    def get_h_hat(self, y, h, x, var, rhh):
        n_r = y.shape[-2]
        n_t = x.shape[-2]
        I = torch.eye(n_t, n_t)
        rhh = conj_t(h) @ h
        h_hat = y @ torch.inverse(conj_t(x) @ rhh @ x + n_r * var * I) @ conj_t(
            x) @ rhh
        return h_hat


class DenoisingMethodModel(DenoisingMethod):

    def __init__(self, model: DenoisingNetModel, use_gpu=True):
        self.model = model
        self.model = self.model.eval()
        self.model.double()
        self.use_gpu = use_gpu and config.USE_GPU
        if self.use_gpu:
            self.model = self.model.cuda()

    def get_key_name(self):
        return self.model.__str__()

    def get_h_hat(self, y, h, x, var, rhh):
        h_ls = y @ torch.inverse(x)
        n_sc = h_ls.shape[1]
        h_ls = h_ls.reshape(-1, *h_ls.shape[-2:])
        h_ls = complex2real(h_ls)
        var = var.repeat(1, n_sc, 1, 1).reshape(-1, 1)
        if self.use_gpu:
            h_ls = h_ls.cuda()
            var = var.cuda()
        h_hat, _ = self.model(h_ls, (var / 2) ** 0.5)
        h_hat = h_hat.reshape(h.shape + (2,))
        h_hat = h_hat[:, :, :, :, 0] + h_hat[:, :, :, :, 1] * 1j
        if h_hat.is_cuda:
            h_hat = h_hat.cpu()
        return h_hat
