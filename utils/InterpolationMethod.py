import abc
import torch

from model import InterpolationNetModel

from utils import complex2real
from utils import line_interpolation_hp_pilot
from utils import get_interpolation_pilot_idx
from utils import DenoisingMethod


class InterpolationMethod(abc.ABC):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__()
        self.n_sc = n_sc
        self.pilot_count = pilot_count
        self.denoisingMethod = denoisingMethod
        self.pilot_idx = get_interpolation_pilot_idx(n_sc, pilot_count)

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_H_hat(self, y, H, xp, var):
        pass

    def get_nmse(self, y, H, xp, var):
        H_hat = self.get_H_hat(y, H, xp, var)
        nmse = ((torch.abs(H - H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(H) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()


class InterpolationMethodLine(InterpolationMethod):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__(n_sc, pilot_count, denoisingMethod)

    def get_key_name(self):
        return 'line'

    def get_H_hat(self, y, H, xp, var):
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            y = y[:, self.pilot_idx]
            h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, None)
        H_hat = line_interpolation_hp_pilot(h_p, self.pilot_idx, self.n_sc)
        return H_hat


class InterpolationMethodModel(InterpolationMethodLine):

    def __init__(self, model: InterpolationNetModel, use_gpu, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__(model.n_sc, model.pilot_count, denoisingMethod)
        self.model = model
        self.model.double()
        self.model.eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = model.cuda()

    def get_key_name(self):
        return self.model.name

    def get_H_hat(self, y, H, xp, var):
        H_shape = H.shape
        H = super().get_H_hat(y, H, xp, var)
        H = H.reshape([H_shape[0], -1, H_shape[-1]])
        H = complex2real(H)
        H = H.permute(0, 3, 1, 2)
        if self.use_gpu:
            H = H.cuda()
        H_hat, = self.model(H, )
        H_hat = H_hat.permute(0, 2, 3, 1)
        H_hat = H_hat[:, :, :, 0] + H_hat[:, :, :, 1] * 1j
        H_hat = H_hat.reshape(H_shape)
        if H_hat.is_cuda:
            H_hat = H_hat.cpu()
        return H_hat
