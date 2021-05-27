import abc
import torch

from model import DetectionNetModel
from utils import complex2real


class DetectionMethod(abc.ABC):

    def __init__(self, constellation):
        self.constellation = constellation

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_x_hat(self, y, h, x, var):
        pass

    def get_nmse(self, y, h, x, var):
        x_hat = self.get_x_hat(y, h, x, var)
        # x = complex2real(x)
        # x_hat = complex2real(x_hat)
        nmse = ((torch.abs(x - x_hat)**2).sum(-1).sum(-1) / (torch.abs(x)**2).sum(-1).sum(-1)).mean()
        nmse = 10*torch.log10(nmse)
        return nmse.item()

    def get_ber(self):
        pass


class DetectionMethodZF(DetectionMethod):

    def __init__(self, constellation):
        super().__init__(constellation)

    def get_key_name(self):
        return 'ZF'

    def get_x_hat(self, y, h, x, var):
        x_hat = torch.inverse(h.conj().transpose(-1, -2) @ h) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodMMSE(DetectionMethod):

    def __init__(self, constellation):
        super().__init__(constellation)

    def get_key_name(self):
        return 'mmse'

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        x_hat = torch.inverse(A) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodModel(DetectionMethod):

    def __init__(self, model: DetectionNetModel, constellation):
        self.model = model.eval()
        super().__init__(constellation)

    def get_key_name(self):
        return self.model.__str__()

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        b = h.conj().transpose(-1, -2) @ y

        b = torch.cat((b.real, b.imag), 2)
        A_left = torch.cat((A.real, A.imag), 2)
        A_right = torch.cat((-A.imag, A.real), 2)
        A = torch.cat((A_left, A_right), 3)

        x_hat, = self.model(A, b)  # reshape???
        x_hat = x_hat[:, :, 0:x.shape[-2], :] + x_hat[:, :, x.shape[-2]:, :] * 1j
        return x_hat
