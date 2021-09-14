import abc
import torch
import torch.nn.functional as F

from loader import CsiDataloader
from model import DetectionNetModel
from utils import complex2real


class DetectionMethod(abc.ABC):

    def __init__(self, modulation):
        self.modulation = modulation

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
        nmse = ((torch.abs(x - x_hat) ** 2).sum(-1).sum(-1) / (torch.abs(x) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()

    def get_ber(self, y, h, x, x_idx, var):
        x_hat = self.get_x_hat(y, h, x, var)
        x_hat_r = x_hat.real
        x_hat_i = x_hat.imag
        constellation = torch.from_numpy(CsiDataloader.constellations[self.modulation])
        constellation_r = constellation.real
        constellation_i = constellation.imag

        x_hat_dist = (x_hat_r - constellation_r) ** 2 + (x_hat_i - constellation_i) ** 2
        x_hat_idx = torch.argmin(x_hat_dist, dim=-1, keepdim=True)
        ber = (x_hat_idx != x_idx).sum() / x_idx.numel()
        return ber.item()


class DetectionMethodZF(DetectionMethod):

    def __init__(self, modulation):
        super().__init__(modulation)

    def get_key_name(self):
        return 'ZF'

    def get_x_hat(self, y, h, x, var):
        x_hat = torch.inverse(h.conj().transpose(-1, -2) @ h) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodMMSE(DetectionMethod):

    def __init__(self, modulation):
        super().__init__(modulation)

    def get_key_name(self):
        return 'mmse'

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        x_hat = torch.inverse(A) @ h.conj().transpose(-1, -2) @ y
        return x_hat


class DetectionMethodModel(DetectionMethod):

    def __init__(self, model: DetectionNetModel, modulation, use_gpu):
        self.model = model.eval()
        self.use_gpu = use_gpu
        super().__init__(modulation)

    def get_key_name(self):
        return self.model.name

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        b = h.conj().transpose(-1, -2) @ y

        b = torch.cat((b.real, b.imag), 2)
        A_left = torch.cat((A.real, A.imag), 2)
        A_right = torch.cat((-A.imag, A.real), 2)
        A = torch.cat((A_left, A_right), 3)
        if self.use_gpu:
            A = A.cuda()
            b = b.cuda()
        x_hat, = self.model(A, b)  # reshape???
        x_hat = x_hat[:, :, 0:x.shape[-2], :] + x_hat[:, :, x.shape[-2]:, :] * 1j
        if x_hat.is_cuda:
            x_hat = x_hat.cpu()
        return x_hat


class DetectionMethodConjugateGradient(DetectionMethod):

    def __init__(self, modulation, iterate, name_add_iterate=True):
        self.iterate = iterate
        self.name_add_iterate = name_add_iterate
        super().__init__(modulation)

    def get_key_name(self):
        if self.name_add_iterate:
            return 'cg-{}th'.format(self.iterate)
        else:
            return self.get_key_name_short()

    def get_key_name_short(self):
        return 'cg'

    @staticmethod
    def conjugate(s, r, d, A):
        alpha = (r.conj().transpose(-1, -2) @ r) / (r.conj().transpose(-1, -2) @ A @ d)
        s_next = s + alpha * d
        r_next = r - alpha * (A @ d)
        beta = (r_next.conj().transpose(-1, -2) @ r_next) / (r.conj().transpose(-1, -2) @ r)
        d_next = r_next + beta * d
        return s_next, r_next, d_next

    def get_x_hat(self, y, h, x, var):
        A = h.conj().transpose(-1, -2) @ h + var * torch.eye(h.shape[-1], h.shape[-1])
        b = h.conj().transpose(-1, -2) @ y

        b = torch.cat((b.real, b.imag), 2)
        A_left = torch.cat((A.real, A.imag), 2)
        A_right = torch.cat((-A.imag, A.real), 2)
        A = torch.cat((A_left, A_right), 3)

        s = torch.zeros(b.shape)
        if torch.cuda.is_available():
            s = s.cuda()
        r = b
        d = r
        for i in range(self.iterate):
            s, r, d = DetectionMethodConjugateGradient.conjugate(s, r, d, A)
        s = s[:, :, 0:x.shape[-2], :] + s[:, :, x.shape[-2]:, :] * 1j
        return s
