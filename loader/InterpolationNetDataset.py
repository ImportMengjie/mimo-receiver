import torch
import random

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType
from utils import DenoisingMethod
from utils import complex2real
from utils import get_interpolation_pilot_idx
from utils import line_interpolation_hp_pilot
from utils import to_cuda
import utils.config as config


class InterpolationNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, pilot_count: int, ) -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.pilot_count = pilot_count
        self.pilot_idx = get_interpolation_pilot_idx(csiDataloader.n_sc, pilot_count)
        self.pilot_count = torch.sum(self.pilot_idx)

        self.xp = csiDataloader.get_pilot_x()
        self.h_p = self.h[:, self.pilot_idx, :, :]
        hx = self.h @ self.xp
        self.var = csiDataloader.get_var_from_snr(hx, snr_range)
        self.hx = self.h_p @ self.xp
        self.xp_inv = torch.inverse(self.xp)

    def cuda(self):
        pass

    def __len__(self):
        return self.h.shape[0] * self.csiDataloader.n_t

    def __getitem__(self, index):
        n_j = index // self.csiDataloader.n_t
        n_t_user = index % self.csiDataloader.n_t
        var = self.var[random.randint(0, self.var.shape[0] - 1), 0, 0, 0]

        H = self.h[n_j]
        y = self.hx[n_j] + self.csiDataloader.get_noise_from_half_sigma((var/2)**0.5, count=self.pilot_count)
        h_pilot_ls = y @ self.xp_inv
        H_interpolation = line_interpolation_hp_pilot(h_pilot_ls.reshape((1, ) + h_pilot_ls.shape), self.pilot_idx, self.csiDataloader.n_sc, True)
        H_interpolation.squeeze_()
        H = complex2real(H[:, :, n_t_user])
        H_interpolation = complex2real(H_interpolation[:, :, n_t_user])
        if config.USE_GPU:
            H = H.cuda()
            H_interpolation = H_interpolation.cuda()
            var = var.cuda()
        return H_interpolation, H, var


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = InterpolationNetDataset(csiDataloader, DataType.train, [100, 101], 3)
    # h, H = dataset.__getitem__(1)
