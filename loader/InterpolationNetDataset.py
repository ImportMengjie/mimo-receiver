import torch
from torch.utils.data.dataset import T_co

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType


class InterpolationNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, pilot_count: int) -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.pilot_count = pilot_count
        self.pilot_idx = []
        count = 0
        for i in range(csiDataloader.n_sc - 1):
            if i % (csiDataloader.n_sc // (pilot_count - 1)) == 0 and count < self.pilot_count:
                count += 1
                self.pilot_idx.append(True)
            else:
                self.pilot_idx.append(False)
        self.pilot_idx.append(True)
        self.pilot_idx = torch.Tensor(self.pilot_idx).bool()
        self.xp = torch.from_numpy(csiDataloader.get_pilot_x())
        self.n = self.n[:, self.pilot_idx, :, :]
        self.h_p = self.h[:, self.pilot_idx, :, :]
        self.y = self.h_p @ self.xp + self.n

        a = 1

    def __len__(self):
        return self.h_p.shape[0]

    def __getitem__(self, index) -> T_co:
        x = self.xp[index]
        y = self.y[index]

        # h need mmse ls model est
        h = self.h_p[index]

        H = self.h[index]

        h = h.reshape([-1, h.shape[-1]])
        H = H.reshape([-1, H.shape[-1]])
        h = BaseDataset.complex2real(h)
        H = BaseDataset.complex2real(H)

        h = h.permute(2, 0, 1)
        H = H.permute(2, 0, 1)
        return h, H


if __name__ == '__main__':
    csiDataloader = CsiDataloader('../data/h_16_16_64_1.mat')
    dataset = InterpolationNetDataset(csiDataloader, DataType.train, [100, 101], 3)
    h, H = dataset.__getitem__(1)
    pass
