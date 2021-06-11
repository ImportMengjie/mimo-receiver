import torch

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType
from utils import complex2real
from utils import to_cuda
import utils.config as config


class DenoisingNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, datatype: DataType, snr_range: list):
        super().__init__(csiDataloader, datatype, snr_range)
        self.x_p = csiDataloader.get_pilot_x()
        hx = self.h @ self.x_p
        self.n, self.sigma = csiDataloader.noise_snr_range(hx, snr_range)
        self.sigma = (self.sigma / 2) ** 0.5
        self.y = hx + self.n
        x_p_inv = torch.inverse(self.x_p)
        self.h_ls = self.y @ x_p_inv
        self.in_cuda = False

    def cuda(self):
        if torch.cuda.is_available():
            self.in_cuda = True
            self.h_ls = self.h_ls.cuda()
            self.h = self.h.cuda()
            self.sigma = self.sigma.cuda()

    def __len__(self):
        return self.h.shape[0] * self.h.shape[1]

    def __getitem__(self, item):
        n_sc_idx = item % self.csiDataloader.n_sc
        idx = item // self.csiDataloader.n_sc
        h = complex2real(self.h[idx, n_sc_idx])
        h_ls = complex2real(self.h_ls[idx, n_sc_idx])
        sigma = self.sigma[idx, 0, 0, 0]
        if config.USE_GPU:
            h_ls = to_cuda(h_ls)
            h = to_cuda(h)
            sigma = to_cuda(sigma)
        return h_ls, h, sigma


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    nmses = []
    snrs = [i for i in range(100, 201, 1)]
    for snr in snrs:
        denoising = DenoisingNetDataset(csiDataloader, DataType.train, [snr, snr + 1])
        h = complex2real(denoising.h)
        h_hat = DenoisingNetDataset.complex2real(denoising.h_ls)

        nmse = (((h_hat - h) ** 2) / (h ** 2)).mean()
        nmse = 10 * torch.log10(nmse)
        # nmse2 = (((h_hat2 - h)**2)/(h**2)).mean()
        # nmse = (((h_hat - h)**2)/(h**2)).mean()
        # nmses.append((nmse, nmse2))
        nmses.append(nmse)

    plt.plot(snrs, nmses)
    plt.show()
