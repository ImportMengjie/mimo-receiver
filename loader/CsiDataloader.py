import logging
from enum import Enum

import h5py
import numpy as np
from h5py import Dataset


class DataType(Enum):
    train = 1
    test = 2


class ChannelType(Enum):
    gaussian = 1
    gpp = 2
    unknown = 3


class CsiDataloader:
    constellations = {
        'qpsk': np.array(
            [complex(1 / np.sqrt(2.), 1 / np.sqrt(2.)), complex(-1 / np.sqrt(2.), 1 / np.sqrt(2.)),
             complex(1 / np.sqrt(2.), -1 / np.sqrt(2.)), complex(-1 / np.sqrt(2.), -1 / np.sqrt(2.))]),
    }

    @staticmethod
    def complex2real(h_mtx):
        h_real = np.real(h_mtx).reshape(h_mtx.shape + (1,))
        h_imag = np.imag(h_mtx).reshape(h_mtx.shape + (1,))
        return np.concatenate((h_real, h_imag), axis=len(h_real.shape) - 1)

    @staticmethod
    def real2complex(h_mtx):
        if len(h_mtx.shape) == 5:
            h_real = h_mtx[:, :, :, :, 0].reshape(h_mtx.shape[:-1])
            h_imag = h_mtx[:, :, :, :, 1].reshape(h_mtx.shape[:-1])
            h_mtx_complex = h_real + 1j * h_imag
            return h_mtx_complex
        else:
            raise Exception('real to complex fail, h shape:{}.'.format(h_mtx.shape))

    def __init__(self, path, train_data_radio=0.9):
        self.path = path
        self.train_data_radio = train_data_radio
        self.channel_type = ChannelType.unknown
        for t in ChannelType:
            if t.name in path:
                self.channel_type = t
        logging.info('loading {}'.format(path))
        files = h5py.File(path, 'r')
        H = files.get('H')
        if type(H) is not Dataset:
            H = H.get("value")
        H = np.array(H).transpose()
        data = CsiDataloader.real2complex(H)
        files.close()

        self.J = data.shape[0]
        self.n_c = 1
        self.n_sc = data.shape[1]
        self.n_r = data.shape[2]
        self.n_t = data.shape[3]
        logging.info('loaded J={},n_c={},n_r={},n_t={},n_sc={}'.format(self.J, self.n_c, self.n_r, self.n_t, self.n_sc))

        self.train_count = int(self.train_data_radio * self.J) * self.n_c
        data = data.reshape(self.J * self.n_c, self.n_sc, self.n_r, self.n_t)
        self.train_H = data[:self.train_count]
        self.test_H = data[self.train_count:]

    def noise_snr_range(self, hx: np.ndarray, snr_range: list, one_col=False):
        count = hx.shape[0]
        snrs = np.random.randint(snr_range[0], snr_range[1], (count, 1))
        if self.channel_type == ChannelType.gpp or one_col == False:
            hx_mean = (CsiDataloader.complex2real(hx) ** 2).mean(-1).mean(-1).mean(-1).mean(-1).reshape(-1, 1)
            noise_var = hx_mean * np.power(10, -snrs / 10.)
        else:
            noise_var = self.n_t / self.n_r * np.power(10, -snrs / 10.)
        n_t = 1 if one_col else self.n_t
        noise_var = (1 / n_t) ** 2 * noise_var
        noise_var = noise_var.reshape(noise_var.shape + (1, 1))
        noise_real = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_sc, self.n_r, n_t])
        noise_imag = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_sc, self.n_r, n_t])
        noise_mat = noise_real + 1j * noise_imag
        return noise_mat, noise_var

    def train_X(self, modulation):
        return self.random_x(self.train_H.shape[0], modulation)

    def test_X(self, modulation):
        return self.random_x(self.test_H.shape[0], modulation)

    def random_x(self, count, modulation):
        constellation_idx_mat = np.random.randint(0, CsiDataloader.constellations[modulation.lower()].shape[0],
                                                  size=(count, self.n_sc, self.n_t, 1))
        return np.array(list(map(lambda x: CsiDataloader.constellations[modulation.lower()][x], constellation_idx_mat)))

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
            return x
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


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    cd = CsiDataloader('../data/h_16_16_64_5.mat')
    # cd = CsiDataloader('../data/h_16_16_64_1.mat')
