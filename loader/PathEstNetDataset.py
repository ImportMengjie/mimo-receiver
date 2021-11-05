import numpy as np
import torch

import utils.config as config
from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType
from utils import DenoisingMethodLS
from utils import complex2real
from utils import to_cuda


class PathEstDnnNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, path_range: list,
                 fix_path=None):
        super().__init__(csiDataloader, dataType, snr_range)
        self.var = None
        self.var_hat = None
        self.g_hat_idft = None
        self.path_range = path_range
        self.xp = csiDataloader.get_pilot_x()
        h = csiDataloader.get_h(dataType=dataType)
        self.hx = h @ self.xp
        self.reload()
        self.fix_path = fix_path
        self.get_path_count = lambda idx: self.fix_path if self.fix_path is not None else self.csiDataloader.path_count[
            idx]
        self.row_count_per_h = self.path_range[1] - self.path_range[0]

    def reload(self):
        n, self.var = self.csiDataloader.noise_snr_range(self.hx, self.snr_range, one_col=False)
        y = self.hx + n
        h_hat = DenoisingMethodLS().get_h_hat(y, self.h, self.xp, self.var, self.csiDataloader.rhh)
        g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape(-1, self.csiDataloader.n_sc, self.csiDataloader.n_r)
        self.g_hat_idft = complex2real(torch.from_numpy(np.fft.ifft(g_hat, axis=-2)))
        chuck_g = self.g_hat_idft[:, self.path_range[1]:, ]
        chuck_g = chuck_g.reshape((chuck_g.shape[0], -1))
        self.var_hat = (chuck_g ** 2).mean(dim=-1)

    def cuda(self):
        pass

    def __len__(self):
        return self.g_hat_idft.shape[0] * self.row_count_per_h

    def __getitem__(self, idx):
        idx_row = idx % self.row_count_per_h + self.path_range[0]
        idx_h = idx // self.row_count_per_h
        right_path = self.get_path_count(idx_h)
        right_y = 1 if right_path >= (idx_row + 1) else 0
        g = self.g_hat_idft[idx_h]
        g_row = g[idx_row].flatten()
        right_y = torch.squeeze(torch.tensor(right_y)).double()
        idx_row = torch.tensor(idx_row)
        right_path = torch.tensor(right_path)
        true_var = self.var[idx_h // self.csiDataloader.n_t].flatten() / 2 / self.csiDataloader.n_sc
        est_var = self.var_hat[idx_h]
        if config.USE_GPU:
            g = to_cuda(g)
            right_y = to_cuda(right_y)
            idx_row = to_cuda(idx_row)
            true_var = to_cuda(true_var)
            right_path = to_cuda(right_path)
            g_row = to_cuda(g_row)
            est_var = to_cuda(est_var)
        return g, idx_row, g_row, right_y, right_path, true_var, est_var


class PathEstCnnNetDataset(BaseDataset):

    def cuda(self):
        pass

    def reload(self):
        n, self.var = self.csiDataloader.noise_snr_range(self.hx, self.snr_range, one_col=False)
        y = self.hx + n
        h_hat = DenoisingMethodLS().get_h_hat(y, self.h, self.xp, self.var, self.csiDataloader.rhh)
        g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape(-1, self.csiDataloader.n_sc, self.csiDataloader.n_r)
        self.g_hat = torch.from_numpy(g_hat)
        self.g_hat_idft = np.fft.ifft(g_hat, axis=-2)
        chuck_g = complex2real(torch.from_numpy(self.g_hat_idft[:, self.path_range[1]:, ]))
        chuck_g = chuck_g.reshape((chuck_g.shape[0], -1))
        self.var_hat = (chuck_g ** 2).mean(-1)

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, path_range: list,
                 fix_path=None):
        super().__init__(csiDataloader, dataType, snr_range)
        self.var = None
        self.var_hat = None
        self.g_hat = None
        self.path_range = path_range
        self.xp = csiDataloader.get_pilot_x()
        h = csiDataloader.get_h(dataType=dataType)
        self.hx = h @ self.xp
        self.fix_path = fix_path
        self.get_path_count = lambda idx: self.fix_path if self.fix_path is not None else self.csiDataloader.path_count[
            idx]
        self.row_count_per_h = self.path_range[1] - self.path_range[0]
        self.reload()

    def __len__(self):
        return self.g_hat_idft.shape[0] * self.row_count_per_h

    def __getitem__(self, idx):
        idx_row = idx % self.row_count_per_h + self.path_range[0]
        idx_h = idx // self.row_count_per_h
        right_path = self.get_path_count(idx_h)
        right_y = 1 if right_path >= (idx_row + 1) else 0
        true_var = self.var[idx_h // self.csiDataloader.n_t].flatten() / 2 * (
                    (self.csiDataloader.n_sc - idx_row) / self.csiDataloader.n_sc)
        est_var = self.var_hat[idx_h] * (self.csiDataloader.n_sc - idx_row)

        g_idft = self.g_hat_idft[idx_h]
        g_hat = self.g_hat[idx_h]
        chuck_array = np.concatenate((np.ones(idx_row), np.zeros(self.csiDataloader.n_sc - idx_row))).reshape(
            (-1, 1))
        g_chuck = np.fft.fft(g_idft * chuck_array, axis=-2)
        g_diff = complex2real(g_hat - torch.from_numpy(g_chuck))

        right_y = torch.squeeze(torch.tensor(right_y)).double()
        idx_row = torch.tensor(idx_row)
        right_path = torch.tensor(right_path)

        if config.USE_GPU:
            g_diff = to_cuda(g_diff)
            idx_row = to_cuda(idx_row)
            right_y = to_cuda(right_y)
            right_path = to_cuda(right_path)
            true_var = to_cuda(true_var)
            est_var = to_cuda(est_var)

        return g_diff, idx_row, right_y, right_path, true_var, est_var
