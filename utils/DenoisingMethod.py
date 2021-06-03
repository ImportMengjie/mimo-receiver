import abc
import torch

from model import DenoisingNetModel
from utils import complex2real
from utils import conj_t


class DenoisingMethod(abc.ABC):

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_h_hat(self, y, h, x, var):
        pass

    def get_nmse(self, y, h, x, var):
        h_hat = self.get_h_hat(y, h, x, var)
        nmse = ((torch.abs(h - h_hat) ** 2).sum(-1).sum(-1) / (torch.abs(h) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()


class DenoisingMethodLS(DenoisingMethod):

    def get_key_name(self):
        return 'LS'

    def get_h_hat(self, y, h, x, var):
        h_ls = y @ torch.inverse(x)
        return h_ls


class DenoisingMethodMMSE(DenoisingMethod):

    def get_key_name(self):
        return 'MMSE'

    def get_h_hat(self, y, h, x, var):
        r_h = conj_t(h) @ h
        n_r = y.shape[-2]
        n_t = x.shape[-2]
        I = torch.eye(n_t, n_t)
        if torch.cuda.is_available():
            I = I.cuda()
        h_hat = y @ torch.inverse(conj_t(x) @ r_h @ x + n_r * var * I) @ conj_t(
            x) @ r_h
        return h_hat


class DenoisingMethodModel(DenoisingMethod):

    def __init__(self, model: DenoisingNetModel):
        self.model = model
        self.model = self.model.eval()
        self.model.double()

    def get_key_name(self):
        return self.model.__str__()

    def get_h_hat(self, y, h, x, var):
        h_ls = y @ torch.inverse(x)
        h_ls = h_ls.reshape(-1, *h_ls.shape[-2:])
        h_ls = complex2real(h_ls)
        h_hat, _ = self.model(h_ls)
        h_hat = h_hat.reshape(h.shape + (2,))
        h_hat = h_hat[:, :, :, :, 0] + h_hat[:, :, :, :, 1] * 1j
        return h_hat
