import logging
from collections import Iterable
from enum import Enum

import h5py
import numpy as np
import torch
from h5py import Dataset


def qam_n_constellation(mod_n):
    n = mod_n
    constellation = np.linspace(int(-np.sqrt(n) + 1), int(np.sqrt(n) - 1), int(np.sqrt(n)))
    alpha = np.sqrt((constellation ** 2).mean())
    constellation /= (alpha * np.sqrt(2))
    array = []
    for i in range(len(constellation)):
        for j in range(len(constellation)):
            array.append(constellation[i] + 1j * constellation[j])
    return np.array(array)


class DataType(Enum):
    train = 1
    test = 2


class ChannelType(Enum):
    gaussian = 1
    gpp = 2
    spatial_ULA = 3
    spatial_UPA = 4
    spatial_mu_ULA = 5
    spatial_mu_UPA = 6
    imt_2020 = 7
    unknown = 8


def toNp(tensor: torch.Tensor):
    return tensor.cpu().detach().numpy()


def USE_GPU(func):
    def wrapper(*args, **kwargs):
        ret = func(*args, **kwargs)
        if CsiDataloader.use_gpu:
            if isinstance(ret, Iterable):
                gpu_ret = []
                for r in ret:
                    gpu_ret.append(r.cuda())
                return gpu_ret
            elif isinstance(ret, torch.Tensor):
                return ret.cuda()
            else:
                raise Exception('unknown ret {}'.format(ret))
        else:
            return ret

    return wrapper


class CsiDataloader:
    constellations = {'qpsk': np.array(
        [complex(1 / np.sqrt(2.), 1 / np.sqrt(2.)), complex(-1 / np.sqrt(2.), 1 / np.sqrt(2.)),
         complex(1 / np.sqrt(2.), -1 / np.sqrt(2.)), complex(-1 / np.sqrt(2.), -1 / np.sqrt(2.))]),
        'bpsk': np.array([complex(1, 0), complex(-1, 0)]), '16qam': qam_n_constellation(16)}
    use_gpu = False

    use_gpu = torch.cuda.is_available() and use_gpu

    @staticmethod
    def complex2real(h_mtx):
        h_real = np.real(h_mtx).reshape(h_mtx.shape + (1,))
        h_imag = np.imag(h_mtx).reshape(h_mtx.shape + (1,))
        return torch.cat((h_real, h_imag), dim=len(h_real.shape) - 1)

    @staticmethod
    def real2complex(h_mtx):
        if len(h_mtx.shape) == 5:
            h_real = h_mtx[:, :, :, :, 0].reshape(h_mtx.shape[:-1])
            h_imag = h_mtx[:, :, :, :, 1].reshape(h_mtx.shape[:-1])
            h_mtx_complex = h_real + 1j * h_imag
            return h_mtx_complex
        else:
            raise Exception('real to complex fail, h shape:{}.'.format(h_mtx.shape))

    def __init__(self, path, train_data_radio=0.9, factor=1):
        assert factor >= 1
        assert 0 <= train_data_radio <= 1
        self.path = path
        self.train_data_radio = train_data_radio
        self.channel_type = ChannelType.unknown
        self.factor = factor
        self.path_count = None
        for t in ChannelType:
            if t.name in path:
                self.channel_type = t
        logging.info('loading {}'.format(path))
        files = h5py.File(path, 'r')
        H = files.get('H')
        self.path_count = files.get('path_count', None)
        if type(H) is not Dataset:
            H = H.get("value")
        if self.path_count is not None:
            if type(self.path_count) is not Dataset:
                self.path_count = self.path_count.get('value')
            self.path_count = np.array(self.path_count)
        H = np.array(H).transpose()
        data = CsiDataloader.real2complex(H)
        files.close()

        data = torch.from_numpy(data)
        if CsiDataloader.use_gpu:
            data = data.cuda()
        data = data.repeat((self.factor, 1, 1, 1))
        self.J = data.shape[0]
        self.n_c = 1
        self.n_sc = data.shape[1]
        self.n_r = data.shape[2]
        self.n_t = data.shape[3]

        self.train_count = int(self.train_data_radio * self.J) * self.n_c
        data = data.reshape(self.J * self.n_c, self.n_sc, self.n_r, self.n_t)
        self.rhh = (data.transpose(-1, -2).conj() @ data).mean(0).mean(0)
        self.train_H = data[:self.train_count]
        self.test_H = data[self.train_count:]

        logging.info(
            'loaded J={},n_c={},n_r={},n_t={},n_sc={},train={},test={}'.format(self.J, self.n_c, self.n_r, self.n_t,
                                                                               self.n_sc, self.train_H.shape[0],
                                                                               self.test_H.shape[0]))

    def _get_path_count(self, j: int, m: int):
        return int(self.path_count[j * self.n_t + m].item()) + 32

    def get_path_count(self, dataType: DataType, j, m):
        if dataType == DataType.train:
            return self._get_path_count(j, m)
        else:
            return self._get_path_count(j + self.train_count, m)

    def gen_chuck_array(self, path_count):
        return np.concatenate((np.ones(path_count), np.zeros(self.n_sc - path_count)))

    def get_chuck_array(self, dataType: DataType):
        chuck_array = np.zeros((self.get_h(dataType).shape[0], self.n_t, self.n_sc, self.n_r))
        for j in range(0, chuck_array.shape[0]):
            for m in range(0, self.n_t):
                chuck_array[j, m] = self.gen_chuck_array(self.get_path_count(dataType, j, m)).reshape((-1, 1))
        return chuck_array

    @USE_GPU
    def noise_snr_range(self, hx: torch.Tensor, snr_range: list, one_col=False):
        count = hx.shape[0]
        n_t = 1 if one_col else self.n_t
        noise_var = self.get_var_from_snr(hx, snr_range)
        noise_real = torch.from_numpy(
            np.random.normal(0, np.sqrt(toNp(noise_var) / 2.), [count, self.n_sc, self.n_r, n_t]))
        noise_imag = torch.from_numpy(
            np.random.normal(0, np.sqrt(toNp(noise_var) / 2.), [count, self.n_sc, self.n_r, n_t]))
        noise_mat = noise_real + 1j * noise_imag
        return noise_mat, noise_var

    def get_var_from_snr(self, hx: torch.Tensor, snr_range: list):
        count = hx.shape[0]
        if hx.is_cuda:
            hx = hx.cpu()
        snrs = torch.randint(snr_range[0], snr_range[1], (count, 1))
        if self.channel_type == ChannelType.gpp or 'spatial' in self.channel_type.name or 'imt' in self.channel_type.name:
            hx_mean = (torch.abs(hx) ** 2).mean(-1).mean(-1).mean(-1).mean(-1).reshape(1, 1)
            noise_var = hx_mean * (10 ** (-snrs / 10.))
        else:
            noise_var = self.n_t / self.n_r * np.power(10, -snrs / 10.)
        noise_var = noise_var.reshape(noise_var.shape + (1, 1))
        return noise_var

    def get_noise_from_half_sigma(self, half_sigma: torch.Tensor, one_col=False, count=None):
        n_t = 1 if one_col else self.n_t
        if count is None:
            noise_real = torch.from_numpy(np.random.normal(0, toNp(half_sigma), [self.n_r, n_t]))
            noise_img = torch.from_numpy(np.random.normal(0, toNp(half_sigma), [self.n_r, n_t]))
        else:
            noise_real = torch.from_numpy(np.random.normal(0, toNp(half_sigma), [count, self.n_r, n_t]))
            noise_img = torch.from_numpy(np.random.normal(0, toNp(half_sigma), [count, self.n_r, n_t]))

        return noise_real + 1j * noise_img

    def train_X(self, modulation):
        return self.random_x(self.train_H.shape[0], modulation)

    def test_X(self, modulation):
        return self.random_x(self.test_H.shape[0], modulation)

    @USE_GPU
    def random_x(self, count, modulation):
        constellation_idx_mat = np.random.randint(0, CsiDataloader.constellations[modulation.lower()].shape[0],
                                                  size=(count, self.n_sc, self.n_t, 1))
        return torch.from_numpy(np.array(list(map(lambda x: CsiDataloader.constellations[modulation.lower()][x],
                                                  constellation_idx_mat)))), torch.from_numpy(constellation_idx_mat)

    @USE_GPU
    def get_pilot_x(self, n_t=None):
        if n_t is None:
            n_t = self.n_t
        if n_t & (n_t - 1) == 0:
            np.zeros((n_t, n_t), dtype=complex)
            x = np.array([[1 + 0j, 1 + 0j], [1j, -1j]]) * (np.sqrt(2) / 2)
            i = 2
            while i < n_t:
                x = np.block([[x, x], [x, -x]]) * (np.sqrt(2) / 2)
                i *= 2
            return torch.from_numpy(x)
        else:
            raise Exception('n_t:{} not 2^n'.format(n_t))

    def get_h_x(self, dataType: DataType, modulation: str):
        if dataType is DataType.train:
            return self.train_H, self.train_X(modulation)
        elif dataType is DataType.test:
            return self.test_H, self.test_X(modulation)
        raise Exception("can't support this type {}".format(dataType))

    def get_h(self, dataType: DataType):
        if dataType is DataType.train:
            return self.train_H
        elif dataType is DataType.test:
            return self.test_H
        raise Exception("can't support this type {}".format(dataType))

    def get_x(self, dataType: DataType, modulation: str):
        if dataType is DataType.train:
            return self.train_X(modulation)
        elif dataType is DataType.test:
            return self.test_X(modulation)
        raise Exception("can't support this type {}".format(dataType))

    def __str__(self):
        return ('n' if 'normal' in self.path else '') + self.channel_type.name


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    cd = CsiDataloader('../data/spatial_mu_ULA_64_32_64_10_l5_11.mat')
    # cd = CsiDataloader('../data/h_16_16_64_1.mat')
