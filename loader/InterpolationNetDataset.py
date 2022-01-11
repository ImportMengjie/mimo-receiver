import logging

import torch

import config.config as config
from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType
from utils import complex2real, InterpolationMethodTransformChuck
from utils import get_interpolation_idx_nf
from utils import to_cuda


class InterpolationNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list,
                 chuckMethod: InterpolationMethodTransformChuck, n_f: int = 0, ) -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.n_f = n_f
        self.pilot_idx = get_interpolation_idx_nf(csiDataloader.n_sc, n_f)
        self.pilot_count = torch.sum(self.pilot_idx)
        self.snr_range = snr_range
        self.chuckMethod = chuckMethod

        self.xp = csiDataloader.get_pilot_x()
        self.hx = self.h @ self.xp
        self.reload()

    def reload(self):
        logging.info('InterpolationNetDataset:reload gen data')
        n, var = self.csiDataloader.noise_snr_range(self.hx, self.snr_range, one_col=False)
        y = self.hx + n
        H_hat, est_left_var_list = self.chuckMethod.get_H_hat_and_var(y, self.h, self.xp, var, self.csiDataloader.rhh)
        self.G_hat = H_hat.permute(0, 3, 1, 2)
        self.est_left_var_list = est_left_var_list
        logging.info('InterpolationNetDataset:reload done')

    def cuda(self):
        pass

    def __len__(self):
        return self.h.shape[0] * self.csiDataloader.n_t

    def __getitem__(self, index):
        n_j = index // self.csiDataloader.n_t
        n_t_user = index % self.csiDataloader.n_t
        g = self.h[n_j, :, :, n_t_user]
        g = complex2real(g)
        g_hat = self.G_hat[n_j, n_t_user]
        g_hat = complex2real(g_hat)
        est_left_var = self.est_left_var_list[n_j, n_t_user, 0, 0]
        if config.USE_GPU:
            g_hat = to_cuda(g_hat)
            g = to_cuda(g)
            est_left_var = to_cuda(est_left_var)
        return g_hat, g, est_left_var


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = InterpolationNetDataset(csiDataloader, DataType.train, [100, 101], 3)
    # h, H = dataset.__getitem__(1)
