import torch.utils.data
import numpy as np
from loader import CsiDataloader
from loader import DataType


class DenoisingNetDataloader(torch.utils.data.DataLoader):

    @staticmethod
    def complex2real(mat: torch.Tensor):
        return torch.cat((mat.real.reshape(mat.shape + (1,)), mat.imag.reshape(mat.shape + (1,))), len(mat.shape))

    def __init__(self, csiDataloader: CsiDataloader, type_: DataType, snr_range: list):
        self.csiDataloader = csiDataloader
        self.type_ = type_
        self.snr_range = snr_range
        self.h = csiDataloader.get_h(type_)
        self.x_p = csiDataloader.get_pilot_x()
        self.n, self.sigma = csiDataloader.noise_snr_range(self.h.shape[0], snr_range)
        self.sigma = np.sqrt(self.sigma)
        self.y = np.matmul(self.h, self.x_p) + self.n
        # self.h_ls = self.y.dot(self.x.conj().T).dot(np.linalg.inv(self.x.dot(self.x.conj().T)))

        self.h_ls = self.y @ np.linalg.inv(self.x_p)

        self.rhh = self.h@self.h.swapaxes(-1, -2).conj()
        self.rhh = self.rhh.mean(axis=0).mean(axis=0)

        self.h_mmse = self.rhh@(self.rhh+self.sigma**2)@self.h_ls

        self.h = torch.from_numpy(self.h)
        self.x_p = torch.from_numpy(self.x_p)
        self.n = torch.from_numpy(self.n)
        self.sigma = torch.from_numpy(self.sigma)
        self.y = torch.from_numpy(self.y)
        self.h_ls = torch.from_numpy(self.h_ls)
        self.h_mmse = torch.from_numpy(self.h_mmse)

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
    import matplotlib.pyplot as plt

    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    nmses = []
    snrs = [i for i in range(40, 61, 1)]
    for snr in snrs:

        denoising = DenoisingNetDataloader(csiDataloader, DataType.train, [snr, snr+1])
        h = DenoisingNetDataloader.complex2real(denoising.h)
        h_hat = DenoisingNetDataloader.complex2real(denoising.h_mmse)
        h_hat2 = DenoisingNetDataloader.complex2real(denoising.h_ls)

        nmse = (((h_hat - h)**2)/(h**2)).mean()
        # nmse2 = (((h_hat2 - h)**2)/(h**2)).mean()
        # nmse = (((h_hat - h)**2)/(h**2)).mean()
        # nmses.append((nmse, nmse2))
        nmses.append(nmse)

    plt.plot(snrs, nmses)
    plt.show()

    pass
