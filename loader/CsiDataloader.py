import logging
from enum import Enum

import h5py
import numpy as np


class DataType(Enum):
    train = 1
    test = 2


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
        if len(h_mtx.shape) == 6:
            h_real = h_mtx[:, :, :, :, :, 0].reshape(h_mtx.shape[:-1])
            h_imag = h_mtx[:, :, :, :, :, 1].reshape(h_mtx.shape[:-1])
            h_mtx_complex = h_real + 1j * h_imag
            return h_mtx_complex
        else:
            raise Exception('real to complex fail, h shape:{}.'.format(h_mtx.shape))

    def __init__(self, path):
        self.path = path
        self.train_data_radio = 0.8
        logging.info('loading {}'.format(path))
        files = h5py.File(path, 'r')
        H = files.get('H')
        self.power_ten = int(np.array(files.get('power_ten'))[0][0])
        data = CsiDataloader.real2complex(np.array(H).transpose())

        self.J = data.shape[0]
        self.n_c = data.shape[1]
        self.n_sc = data.shape[2]
        self.n_r = data.shape[3]
        self.n_t = data.shape[4]
        logging.info('loaded J={},n_c={},n_r={},n_t={},n_sc={}'.format(self.J, self.n_c, self.n_r, self.n_t, self.n_sc))

        self.train_count = int(self.train_data_radio * self.J) * self.n_c
        data = data.reshape(self.J * self.n_c, self.n_sc, self.n_r, self.n_t)
        self.train_H = data[:self.train_count] / np.power(10., self.power_ten)
        self.test_H = data[self.train_count:] / np.power(10., self.power_ten)

    # def noise_n(self, count: int, snr: int):
    #     noise_var = self.n_t / self.n_r * np.power(10, -snr / 10.)
    #     noise_real = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_r, 1])
    #     noise_imag = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_r, 1])
    #     noise_mat = noise_real + 1j * noise_imag
    #     return noise_mat

    def noise_snr_range(self, count: int, snr_range: list):
        snrs = np.random.randint(snr_range[0], snr_range[1], (count // self.n_c, 1))
        noise_var = self.n_t / self.n_r * np.power(10, -snrs / 10.)
        noise_var = np.expand_dims(noise_var.repeat(self.n_c, 0), 1).repeat(self.n_sc, 1)
        noise_real = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_sc, self.n_r]).reshape(count,
                                                                                                        self.n_sc,
                                                                                                        self.n_r, 1)
        noise_imag = np.random.normal(0, np.sqrt(noise_var / 2.), [count, self.n_sc, self.n_r]).reshape(count,
                                                                                                        self.n_sc,
                                                                                                        self.n_r, 1)
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

    def get_h_x(self, dataType: DataType, modulation: str):
        if dataType is DataType.train:
            return self.train_H, self.train_X(modulation)
        elif dataType is DataType.test:
            return self.test_H, self.test_X(modulation)
        raise Exception("can't support this type {}".format(dataType))


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    cd = CsiDataloader('../data/h_16_8_64_56.mat')
