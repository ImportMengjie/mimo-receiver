import torch
from torch.utils.data.dataset import T_co

from loader import BaseDataset
from loader import CsiDataloader
from loader import DataType


class InterpolationNetDataset(BaseDataset):

    def __init__(self, csiDataloader: CsiDataloader, dataType: DataType, snr_range: list, pilot_count: int,
                 interpolation=True) -> None:
        super().__init__(csiDataloader, dataType, snr_range)
        self.pilot_count = pilot_count
        self.interpolation = interpolation
        self.pilot_idx = []
        self.repeat_nums = []
        self.line_factor = [0.]

        count = 0
        for i in range(csiDataloader.n_sc - 1):
            if i % (csiDataloader.n_sc // (pilot_count - 1)) == 0 and count < self.pilot_count:
                count += 1
                self.pilot_idx.append(True)
            else:
                self.pilot_idx.append(False)
        self.pilot_idx.append(True)

        left = 0
        for i in range(1, csiDataloader.n_sc):
            if self.pilot_idx[i]:
                self.repeat_nums.append(i - left)
                step = i - left
                self.line_factor = self.line_factor + [j / step for j in range(1, step)]
                self.line_factor.append(0.)
                left = i

        self.pilot_idx = torch.Tensor(self.pilot_idx).bool()
        self.repeat_nums = torch.Tensor(self.repeat_nums)
        self.line_factor = torch.Tensor(self.line_factor).reshape((csiDataloader.n_sc, 1, 1))
        self.xp = torch.from_numpy(csiDataloader.get_pilot_x())
        self.n = self.n[:, self.pilot_idx, :, :]
        self.h_p = self.h[:, self.pilot_idx, :, :]
        self.y = self.h_p @ self.xp + self.n

    def __len__(self):
        return self.h_p.shape[0]

    def __getitem__(self, index) -> T_co:
        x = self.xp
        y = self.y[index]

        # h need mmse ls model est
        h = self.h_p[index]

        H = self.h[index]

        if self.interpolation:
            repeat_h = h[0:1, :, :].repeat(int(self.repeat_nums[0].item()), 1, 1)
            for i in range(1, self.repeat_nums.shape[0]):
                repeat_h = torch.cat((repeat_h, h[i:i + 1, :, :].repeat(int(self.repeat_nums[i].item()), 1, 1)))
            repeat_h = torch.cat((repeat_h, h[-1:]))
            diff_h = None
            for i in range(self.repeat_nums.shape[0]):
                diff = (h[i + 1:i + 2] - h[i:i + 1]).repeat(int(self.repeat_nums[i].item()), 1, 1)
                if diff_h is not None:
                    diff_h = torch.cat((diff_h, diff))
                else:
                    diff_h = diff
            diff_h = torch.cat((diff_h, h[-1:]))
            h = repeat_h + self.line_factor * diff_h

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
