import abc
import torch

from model import CBDNetSFModel

from utils import complex2real, DenoisingMethodLS
from utils import line_interpolation_hp_pilot
from utils import get_interpolation_pilot_idx
from utils import DenoisingMethod
from utils import config


class InterpolationMethod(abc.ABC):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__()
        self.n_sc = n_sc
        self.denoisingMethod = denoisingMethod
        self.pilot_idx = get_interpolation_pilot_idx(n_sc, pilot_count)
        self.pilot_count = torch.sum(self.pilot_idx).item()

    @abc.abstractmethod
    def get_key_name(self):
        pass

    @abc.abstractmethod
    def get_pilot_name(self):
        pass

    @abc.abstractmethod
    def get_H_hat(self, y, H, xp, var, rhh):
        pass

    def get_nmse(self, y, H, xp, var, rhh):
        H_hat = self.get_H_hat(y, H, xp, var, rhh)
        nmse = ((torch.abs(H - H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(H) ** 2).sum(-1).sum(-1)).mean()
        nmse = 10 * torch.log10(nmse)
        return nmse.item()

    def get_pilot_nmse_and_interp_nmse(self, y, H, xp, var, rhh):
        H_hat = self.get_H_hat(y, H, xp, var, rhh)
        pilot_H = H[:, self.pilot_idx]
        pilot_H_hat = H_hat[:, self.pilot_idx]
        data_H = H[:, torch.logical_not(self.pilot_idx)]
        data_H_hat = H_hat[:, torch.logical_not(self.pilot_idx)]
        nmse_pilot = None
        if self.denoisingMethod:
            nmse_pilot = (
                    (torch.abs(pilot_H - pilot_H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(pilot_H) ** 2).sum(-1).sum(
                -1)).mean()
            nmse_pilot = 10 * torch.log10(nmse_pilot)
        nmse_data = ((torch.abs(data_H - data_H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(data_H) ** 2).sum(-1).sum(
            -1)).mean()
        nmse_data = 10 * torch.log10(nmse_data)
        return nmse_pilot, nmse_data


class InterpolationMethodLine(InterpolationMethod):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__(n_sc, pilot_count, denoisingMethod)

    def get_key_name(self):
        if self.denoisingMethod:
            return 'line' + '-' + self.denoisingMethod.get_key_name()
        else:
            return 'line' + '-' + 'true'

    def get_pilot_name(self):
        if self.denoisingMethod:
            return self.denoisingMethod.get_key_name()
        else:
            return 'true'

    def get_H_hat(self, y, H, xp, var, rhh):
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            y = y[:, self.pilot_idx]
            h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, rhh)
        H_hat = line_interpolation_hp_pilot(h_p, self.pilot_idx, self.n_sc, False)
        return H_hat


class InterpolationMethodModel(InterpolationMethodLine):

    def __init__(self, model: CBDNetSFModel, use_gpu, pilot_count=None) -> None:
        denoisingMethod = DenoisingMethodLS()
        if pilot_count is None:
            pilot_count = model.pilot_count
        super().__init__(model.n_sc, pilot_count, denoisingMethod)
        self.model = model.double().eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = model.cuda()

    def get_key_name(self):
        return self.model.name

    def get_pilot_name(self):
        return self.model.name

    def get_H_hat(self, y, H, xp, var, rhh):
        J, n_sc, n_r, n_t = H.shape
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            y = y[:, self.pilot_idx]
            h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, rhh)
        H_hat = line_interpolation_hp_pilot(h_p, self.pilot_idx, self.n_sc)
        H_hat = H_hat.permute(0, 3, 1, 2)
        H_hat = complex2real(H_hat.reshape((-1,) + H_hat.shape[-2:]))
        var = var.repeat((1, 1, 1, n_t)).reshape(-1, 1)
        model_H_hat = None
        for i in range(0, H_hat.shape[0], config.ANALYSIS_BATCH_SIZE):
            H_hat_batch = H_hat[i:i + config.ANALYSIS_BATCH_SIZE]
            var_batch = var[i: i + config.ANALYSIS_BATCH_SIZE]

            if self.use_gpu:
                H_hat_batch = H_hat_batch.cuda()
                var_batch = var_batch.cuda()
            model_H_hat_batch, _ = self.model(H_hat_batch, var_batch)
            if model_H_hat_batch.is_cuda:
                model_H_hat_batch = model_H_hat_batch.detach().cpu()
            if model_H_hat is None:
                model_H_hat = model_H_hat_batch
            else:
                model_H_hat = torch.cat((model_H_hat, model_H_hat_batch), 0)
        model_H_hat = model_H_hat.reshape(J, n_t, n_sc, n_r, 2)
        model_H_hat = model_H_hat.permute(0, 2, 3, 1, 4)
        model_H_hat = model_H_hat[:, :, :, :, 0] + 1j * model_H_hat[:, :, :, :, 1]
        return model_H_hat.detach()
