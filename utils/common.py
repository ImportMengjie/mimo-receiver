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
