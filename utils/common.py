import torch


class AvgLoss:

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0
        self.avg = 0

    def add(self, num):
        self.sum += num
        self.count += 1
        self.avg = self.sum / self.count


def complex2real(mat: torch.Tensor):
    return torch.cat((mat.real.reshape(mat.shape + (1,)), mat.imag.reshape(mat.shape + (1,))), len(mat.shape))


def conj_t(mat: torch.Tensor):
    return mat.conj().transpose(-1, -2)


def line_interpolation_hp_pilot(h_p: torch.Tensor, pilot_idx: torch.Tensor, n_sc: int):
    repeat_nums = []
    line_factor = [0.]
    left = 0
    for i in range(1, n_sc):
        if pilot_idx[i]:
            repeat_nums.append(i - left)
            step = i - left
            line_factor = line_factor + [j / step for j in range(1, step)]
            line_factor.append(0.)
            left = i

    repeat_nums = torch.Tensor(repeat_nums)
    line_factor = torch.Tensor(line_factor).reshape((n_sc, 1, 1))

    repeat_h = h_p[:, 0:1, :, :].repeat(1, int(repeat_nums[0].item()), 1, 1)
    for i in range(1, repeat_nums.shape[0]):
        repeat_h = torch.cat((repeat_h, h_p[:, i:i + 1, :, :].repeat(1, int(repeat_nums[i].item()), 1, 1)), 1)
    repeat_h = torch.cat((repeat_h, h_p[:, -1:]), 1)
    diff_h = None
    for i in range(repeat_nums.shape[0]):
        diff = (h_p[:, i + 1:i + 2] - h_p[:, i:i + 1]).repeat(1, int(repeat_nums[i].item()), 1, 1)
        if diff_h is not None:
            diff_h = torch.cat((diff_h, diff), 1)
        else:
            diff_h = diff
    diff_h = torch.cat((diff_h, h_p[:, -1:]), 1)
    h_interpolation = repeat_h + line_factor * diff_h
    return h_interpolation


def get_interpolation_pilot_idx(n_sc: int, pilot_count: int):
    pilot_idx = []
    count = 0
    for i in range(n_sc - 1):
        if i % (n_sc // (pilot_count - 1)) == 0 and count < pilot_count:
            count += 1
            pilot_idx.append(True)
        else:
            pilot_idx.append(False)
    pilot_idx.append(True)
    pilot_idx = torch.Tensor(pilot_idx).bool()
    return pilot_idx


if __name__ == '__main__':
    h = torch.arange(0, 64).reshape(-1, 1, 1) * torch.ones(8, 8)
    h = h.reshape(1, -1, 8, 8).repeat(10, 1, 1, 1)
    pilot_idx = get_interpolation_pilot_idx(64, 10)
    new_h = line_interpolation_hp_pilot(h[:, pilot_idx], pilot_idx, 64)
    print((new_h - h).sum())
    print(h.shape)
