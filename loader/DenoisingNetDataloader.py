import torch.utils.data
import numpy as np
from loader import CsiDataloader
from loader import DataType


class DenoisingNetDataloader(torch.utils.data.DataLoader):

    @staticmethod
    def complex2real(mat: torch.Tensor):
        return torch.cat((mat.real.reshape(mat.shape + (1,)), mat.imag.reshape(mat.shape + (1,))), len(mat.shape) - 1)

    def __init__(self, csiDataloader: CsiDataloader, type_: DataType, snr_range: list, modulation='qpsk'):
        self.csiDataloader = csiDataloader
        self.type_ = type_
        self.snr_range = snr_range
        self.h, self.x = csiDataloader.get_h_x(type_, modulation)
        self.n, self.sigma = csiDataloader.noise_snr_range(self.h.shape[0], snr_range)
        self.sigma = np.sqrt(self.sigma)
        self.y = np.matmul(self.h, self.x) + self.n
        # self.h_ls = self.y.dot(self.x.conj().T).dot(np.linalg.inv(self.x.dot(self.x.conj().T)))

        self.h_ls = self.y @ self.x.conj().swapaxes(-1, -2) @ np.linalg.pinv(self.x @ self.x.conj().swapaxes(-1, -2))

        self.h = torch.from_numpy(self.h)
        self.x = torch.from_numpy(self.x)
        self.n = torch.from_numpy(self.n)
        self.sigma = torch.from_numpy(self.sigma)
        self.y = torch.from_numpy(self.y)
        self.h_ls = torch.from_numpy(self.h_ls)

    def __len__(self):
        return self.h.shape[0] * self.h.shape[1]

    def __getitem__(self, item):
        n_sc_idx = item % csiDataloader.n_sc
        idx = item // csiDataloader.n_sc
        h = DenoisingNetDataloader.complex2real(self.h[idx, n_sc_idx])
        h_ls = DenoisingNetDataloader.complex2real(self.h_ls[idx, n_sc_idx])
        sigma_map = torch.full((self.csiDataloader.n_r, self.csiDataloader.n_t), self.sigma[idx, n_sc_idx, 0])
        return h, sigma_map, h_ls


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_8_64_56.mat')
    denoising = DenoisingNetDataloader(csiDataloader, DataType.train, [100, 101])
    h, sigma_map, h_ls = denoising[200]
    pass
