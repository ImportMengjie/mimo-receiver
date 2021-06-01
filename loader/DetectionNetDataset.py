import torch

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType


class DetectionNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, modulation='qpsk') -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.csiDataloader = csiDataloader
        self.modulation = modulation
        self.dataType = dataType
        self.x = csiDataloader.get_x(dataType, modulation)
        self.hx = self.h @ self.x
        self.n, self.var = csiDataloader.noise_snr_range(self.hx, snr_range, True)
        self.y = self.hx + self.n
        self.A = self.h.conj().transpose(-1, -2) @ self.h + self.var * torch.eye(csiDataloader.n_t, csiDataloader.n_t)
        self.b = self.h.conj().transpose(-1, -2) @ self.y

        self.x = torch.cat((self.x.real, self.x.imag), 2)
        self.b = torch.cat((self.b.real, self.b.imag), 2)
        A_left = torch.cat((self.A.real, self.A.imag), 2)
        A_right = torch.cat((-self.A.imag, self.A.real), 2)
        self.A = torch.cat((A_left, A_right), 3)

    def cuda(self):
        if torch.cuda.is_available():
            self.A = self.A.cuda()
            self.b = self.b.cuda()
            self.x = self.x.cuda()

    def __len__(self):
        return self.h.shape[0] * self.h.shape[1]

    def __getitem__(self, item):
        n_sc_idx = item % self.csiDataloader.n_sc
        idx = item // self.csiDataloader.n_sc
        A = self.A[idx, n_sc_idx]
        b = self.b[idx, n_sc_idx]
        x = self.x[idx, n_sc_idx]
        return A, b, x


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = DetectionNetDataset(csiDataloader, DataType.train, [10, 101])
    A, b, x = dataset.__getitem__(1)
    print(A.shape, b.shape, x.shape)
