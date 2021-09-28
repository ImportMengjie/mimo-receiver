import abc
import torch
import numpy as np

from model import CBDNetSFModel

from utils import complex2real, DenoisingMethodLS
from utils import line_interpolation_hp_pilot
from utils import get_interpolation_pilot_idx
from utils import DenoisingMethod
from utils import config


class InterpolationMethod(abc.ABC):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None, only_est_data=False,
                 extra='') -> None:
        assert not (denoisingMethod is None and self.only_est_data)
        super().__init__()
        self.n_sc = n_sc
        self.denoisingMethod = denoisingMethod
        self.pilot_idx = get_interpolation_pilot_idx(n_sc, pilot_count)
        self.pilot_count = torch.sum(self.pilot_idx).item()
        self.only_est_data = only_est_data
        self.extra = extra
        self.is_denosing = self.pilot_count == self.n_sc

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
        if self.denoisingMethod and not self.only_est_data:
            nmse_pilot = (
                    (torch.abs(pilot_H - pilot_H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(pilot_H) ** 2).sum(-1).sum(
                -1)).mean()
            nmse_pilot = 10 * torch.log10(nmse_pilot)
        nmse_data = ((torch.abs(data_H - data_H_hat) ** 2).sum(-1).sum(-1) / (torch.abs(data_H) ** 2).sum(-1).sum(
            -1)).mean()
        nmse_data = 10 * torch.log10(nmse_data)
        return nmse_pilot, nmse_data


class InterpolationMethodLine(InterpolationMethod):

    def __init__(self, n_sc, pilot_count: int, denoisingMethod: DenoisingMethod = None, only_data_est=False,
                 extra='') -> None:
        super().__init__(n_sc, pilot_count, denoisingMethod, only_data_est, extra)

    def get_key_name(self):
        if self.denoisingMethod:
            if self.only_est_data:
                key_name = self.denoisingMethod.get_key_name()
            else:
                key_name = 'line' + '-' + self.denoisingMethod.get_key_name()
        else:
            key_name = 'line' + '-' + 'true'
        return key_name + self.extra

    def get_pilot_name(self):
        if self.denoisingMethod:
            return self.denoisingMethod.get_key_name()
        else:
            return 'true'

    def get_H_hat(self, y, H, xp, var, rhh):
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            if self.only_est_data:
                return self.denoisingMethod.get_h_hat(y, H, xp, var, rhh)
            else:
                y = y[:, self.pilot_idx]
                h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, rhh)
        H_hat = line_interpolation_hp_pilot(h_p, self.pilot_idx, self.n_sc, False)
        return H_hat


class InterpolationMethodChuck(InterpolationMethodLine):

    def __init__(self, n_sc, pilot_count: int, path_count: int, denoisingMethod: DenoisingMethod = None,
                 padding_chuck=False, extra=''):
        super().__init__(n_sc, pilot_count, denoisingMethod, False, extra=extra)
        self.chuck_array = np.concatenate((np.ones(path_count), np.zeros(n_sc - path_count)))
        self.chuck_array = self.chuck_array.reshape((-1, 1))
        self.padding_chuck = padding_chuck

    def get_key_name(self):
        if self.is_denosing:
            key_name = 'dft-chuck'
        else:
            key_name = 'dft-padding'
            if self.padding_chuck:
                key_name = key_name + '-chuck'

        if self.denoisingMethod:
            key_name += '-' + self.denoisingMethod.get_key_name()
        return key_name + self.extra

    def get_pilot_name(self):
        if self.denoisingMethod:
            if self.is_denosing:
                ret_name = 'dft-chuck-'
            else:
                ret_name = 'dft-padding-'

            return ret_name + self.denoisingMethod.get_key_name()
        else:
            return 'chuck-true'

    def get_H_hat(self, y, H, xp, var, rhh):
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            y = y[:, self.pilot_idx]
            h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, rhh)
        if self.n_sc == self.pilot_count:
            H_hat = h_p
            H_hat = H_hat.permute(0, 3, 1, 2)
            H_hat = H_hat.numpy()
            H_hat_in_time = np.fft.ifft2(H_hat)
            H_hat_in_time = H_hat_in_time * self.chuck_array
            H_hat = np.fft.fft2(H_hat_in_time)
        else:
            h_p = h_p.permute(0, 3, 1, 2)
            h_p = h_p.numpy()
            h_p_in_time = np.fft.ifft(h_p, axis=-2)
            # h_p_in_time = np.fft.ifft2(h_p)
            split_idx = self.pilot_count
            zeros_count = self.n_sc - self.pilot_count
            H_p_in_time = np.concatenate((h_p_in_time[:, :, :split_idx],
                                          np.zeros((h_p.shape[:2] + (zeros_count, h_p.shape[-1]))),
                                          h_p_in_time[:, :, split_idx:]), axis=-2)
            if self.padding_chuck:
                H_p_in_time = H_p_in_time * self.chuck_array
            H_hat = np.fft.fft(H_p_in_time, axis=-2)

            # H_hat_in_time = np.fft.ifft2(H_hat)
            # H_hat_in_time = H_hat_in_time * self.chuck_array
            # H_hat = np.fft.fft2(H_hat_in_time)

        H_hat = torch.from_numpy(H_hat)
        H_hat = H_hat.permute(0, 2, 3, 1)
        return H_hat


class InterpolationMethodModel(InterpolationMethodLine):

    def __init__(self, model: CBDNetSFModel, use_gpu, pilot_count=None, extra='') -> None:
        denoisingMethod = DenoisingMethodLS()
        if pilot_count is None:
            pilot_count = model.pilot_count
        super().__init__(model.n_sc, pilot_count, denoisingMethod, False, extra)
        self.model = model.double().eval()
        self.use_gpu = use_gpu
        if self.use_gpu:
            self.model = model.cuda()

    def get_key_name(self):
        return self.model.name + self.extra

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

    def get_sigma_hat(self, y, H, xp, var, rhh):
        J, n_sc, n_r, n_t = H.shape
        h_p = H[:, self.pilot_idx]
        if self.denoisingMethod is not None:
            y = y[:, self.pilot_idx]
            h_p = self.denoisingMethod.get_h_hat(y, h_p, xp, var, rhh)
        H_hat = line_interpolation_hp_pilot(h_p, self.pilot_idx, self.n_sc)
        H_hat = H_hat.permute(0, 3, 1, 2)
        H_hat = complex2real(H_hat.reshape((-1,) + H_hat.shape[-2:]))
        var = var.repeat((1, 1, 1, n_t)).reshape(-1, 1)
        sigma_list = []
        for i in range(0, H_hat.shape[0], config.ANALYSIS_BATCH_SIZE):
            H_hat_batch = H_hat[i:i + config.ANALYSIS_BATCH_SIZE]
            var_batch = var[i: i + config.ANALYSIS_BATCH_SIZE]

            if self.use_gpu:
                H_hat_batch = H_hat_batch.cuda()
                var_batch = var_batch.cuda()
            _, var_hat = self.model(H_hat_batch, var_batch)
            if var_hat.is_cuda:
                var_hat = var_hat.cpu()
            sigma_list.extend([s.item() ** 0.5 for s in var_hat.flatten()])
        return sigma_list
