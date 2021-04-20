import torch
from torch.utils.data.dataset import T_co

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType
from utils import complex2real
from utils import line_interpolation_hp_pilot
from utils import get_interpolation_pilot_idx
from utils import DenoisingMethod


class InterpolationNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, pilot_count: int,
                 interpolation=True, denoisingMethod: DenoisingMethod = None) -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.pilot_count = pilot_count
        self.interpolation = interpolation
        self.pilot_idx = get_interpolation_pilot_idx(csiDataloader.n_sc, pilot_count)

        self.xp = torch.from_numpy(csiDataloader.get_pilot_x())
        self.n = self.n[:, self.pilot_idx, :, :]
        self.h_p = self.h[:, self.pilot_idx, :, :]
        self.y = self.h_p @ self.xp + self.n
        if denoisingMethod is not None:
            self.h_p = denoisingMethod.get_h_hat(self.y, self.h_p, self.xp, self.sigma**2)
        self.h_interpolation = None
        if self.interpolation:
            self.h_interpolation = line_interpolation_hp_pilot(self.h_p, self.pilot_idx, csiDataloader.n_sc)

    def __len__(self):
        return self.h_p.shape[0]

    def __getitem__(self, index) -> T_co:
        h = self.h_p[index] if not self.interpolation else self.h_interpolation[index]

        H = self.h[index]

        h = h.reshape([-1, h.shape[-1]])
        H = H.reshape([-1, H.shape[-1]])
        h = complex2real(h)
        H = complex2real(H)

        h = h.permute(2, 0, 1)
        H = H.permute(2, 0, 1)
        return h, H


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = InterpolationNetDataset(csiDataloader, DataType.train, [100, 101], 3)
    h, H = dataset.__getitem__(1)
    pass
