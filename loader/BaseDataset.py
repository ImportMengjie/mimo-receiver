from abc import ABC

import numpy as np
import torch.utils.data

from loader import CsiDataloader
from loader import DataType


class BaseDataset(torch.utils.data.Dataset, ABC):

    @staticmethod
    def complex2real(mat: torch.Tensor):
        return torch.cat((mat.real.reshape(mat.shape + (1,)), mat.imag.reshape(mat.shape + (1,))), len(mat.shape))

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list) -> None:
        super(BaseDataset, self).__init__()
        self.csiDataloader = csiDataloader
        self.dataType = dataType
        self.snr_range = snr_range
        self.h = csiDataloader.get_h(dataType)
        self.n, self.sigma = csiDataloader.noise_snr_range(self.h.shape[0], snr_range)
        self.sigma = np.sqrt(self.sigma)

        self.h = torch.from_numpy(self.h)
        self.n = torch.from_numpy(self.n)
        self.sigma = torch.from_numpy(self.sigma)
