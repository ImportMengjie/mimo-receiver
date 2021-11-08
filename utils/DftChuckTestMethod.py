import abc
import logging

import numpy as np
import scipy.stats
import torch

from model import PathEstBaseModel
from utils import TestMethod

import config

use_gpu = config.USE_GPU and True


class DftChuckTestMethod(abc.ABC):

    def __init__(self, n_r, n_sc, cp, a=0.05, use_true_var=False, testMethod: TestMethod = TestMethod.one_row,
                 full_name=False, min_path=0):
        self.testMethod = testMethod
        self.n_r = n_r
        self.n_sc = n_sc
        self.cp = cp
        self.a = a
        self.not_cp = n_r - cp
        self.use_true_var = use_true_var
        self.full_name = full_name
        self.first = True
        self.update_est_var = False
        self.min_path = min_path

    def get_path_count(self, g_hat_idft: np.ndarray, g_hat, true_var) -> (int, any):
        probability_list = []
        if self.testMethod == TestMethod.one_row:
            g_hat_idft = np.concatenate((g_hat_idft.real, g_hat_idft.imag), axis=1)
            est_var = (g_hat_idft[self.cp:] ** 2).mean()
            true_var = true_var / self.n_sc
            for i in range(self.cp - 1, self.min_path, -1):
                idft_row = g_hat_idft[i].flatten()
                if self.update_est_var:
                    est_var = (g_hat_idft[i + 1:] ** 2).mean()
                is_path, probability = self.test_one_row(idft_row, est_var, true_var, i, g_hat_idft)
                probability_list.append(probability)
                if is_path:
                    return i + 1, probability_list
        elif self.testMethod == TestMethod.whole_noise:
            g_hat_idft = np.concatenate((g_hat_idft.real, g_hat_idft.imag), axis=1)
            est_var = (g_hat_idft[self.cp:] ** 2).mean()
            true_var = true_var / self.n_sc
            for i in range(self.cp - 1, self.min_path, -1):
                if self.update_est_var:
                    est_var = (g_hat_idft[i + 1:] ** 2).mean()
                idft_rows = g_hat_idft[i:].flatten()
                is_path, probability = self.test_whole_noise(idft_rows, est_var, true_var, i, g_hat_idft)
                probability_list.append(probability)
                if is_path:
                    return i + 1, probability_list
        elif self.testMethod == TestMethod.dft_diff:
            chuck_array = np.concatenate((np.ones(self.cp), np.zeros(self.n_sc - self.cp))).reshape((-1, 1))
            est_cp_var = (np.abs(g_hat_idft[self.cp:]) ** 2).mean()
            true_var_o = true_var
            for i in range(self.cp - 1, self.min_path, -1):
                if self.update_est_var:
                    est_var = (np.abs(g_hat_idft[i + 1:]) ** 2).mean() * (self.n_sc - i) / 2
                else:
                    est_var = est_cp_var * (self.n_sc - i) / 2
                true_var = true_var_o * (self.n_sc - i) / self.n_sc
                chuck_array[i] = 0
                idft_g_chuck = g_hat_idft * chuck_array
                g_chuck = np.fft.fft(idft_g_chuck, axis=0)
                dft_diff = g_hat - g_chuck
                dft_diff = np.concatenate(
                    [dft_diff.real.reshape(dft_diff.shape + (1,)), dft_diff.imag.reshape(dft_diff.shape + (1,))], -1)
                is_path, probability = self.test_dft_diff(dft_diff, est_var, true_var, i, g_chuck)
                probability_list.append(probability)
                if is_path:
                    return i + 1, probability_list
        if self.first:
            logging.warning('get zero path in {}'.format(self.name()))
            self.first = False
        return self.min_path, probability_list

    @abc.abstractmethod
    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        pass

    @abc.abstractmethod
    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        pass

    @abc.abstractmethod
    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        pass

    @abc.abstractmethod
    def name(self):
        return 'abs_dft_chuck'


class ModelPathestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        idft_g = torch.from_numpy(idft_g.reshape((1,) + idft_g.shape))
        idft_row = torch.from_numpy(idft_row.reshape((1,) + idft_row.shape))
        true_var = torch.tensor(true_var)
        est_var = torch.tensor(est_var)
        if use_gpu:
            idft_g = idft_g.cuda()
            idft_row = idft_row.cuda()
            true_var = true_var.cuda()
            est_var = est_var.cuda()
        p, = self.model(idft_g, 0, idft_row, true_var, est_var)
        return p >= 0.5, p

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        raise Exception('not impl test_whole_noise')

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        dft_diff = torch.from_numpy(dft_diff.reshape((1,) + dft_diff.shape))
        true_var = torch.tensor(true_var)
        est_var = torch.tensor(est_var)
        if use_gpu:
            dft_diff = dft_diff.cuda()
            true_var = true_var.cuda()
            est_var = est_var.cuda()
        p, = self.model(dft_diff, 0, true_var, est_var)
        return p >= 0.5, p

    def name(self):
        return self.model.name

    def __init__(self, n_r, n_sc, cp, model: PathEstBaseModel, use_true_var=False, full_name=False):
        super().__init__(n_r, n_sc, cp, 0.05, use_true_var, model.test_method, full_name)
        self.model = model
        model.use_true_var = use_true_var


class VarTestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        var = true_var if self.use_true_var else est_var
        cur_row_chi = (len(idft_row) - 1) * np.var(idft_row) / var
        p = 1 - scipy.stats.chi2(len(idft_row) - 1).cdf(cur_row_chi)
        return p <= self.a, p

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        var = true_var if self.use_true_var else est_var
        cur_row_chi = (len(idft_rows) - 1) * np.var(idft_rows) / var
        p = 1 - scipy.stats.chi2(len(idft_rows) - 1).cdf(cur_row_chi)
        return p <= self.a, p

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        dft_diff = dft_diff.flatten()
        var = true_var if self.use_true_var else est_var
        cur_row_chi = (len(dft_diff) - 1) * np.var(dft_diff) / var
        p = 1 - scipy.stats.chi2(len(dft_diff) - 1).cdf(cur_row_chi)
        return p <= self.a, p

    def __init__(self, n_r, n_sc, cp, a=0.05, use_true_var=False, testMethod: TestMethod = TestMethod.one_row,
                 full_name=False):
        super().__init__(n_r, n_sc, cp, a=a, use_true_var=use_true_var, testMethod=testMethod, full_name=full_name)

    def name(self):
        return 'var-test-{}'.format(self.testMethod.name)


class SWTestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        w, p = scipy.stats.shapiro(idft_row)
        return p <= self.a, p

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        w, p = scipy.stats.shapiro(idft_rows)
        return p <= self.a, p

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        w, p = scipy.stats.shapiro(dft_diff.flatten())
        return p <= self.a, p

    def __init__(self, n_r, n_sc, cp, a=0.05, use_true_var=False, testMethod: TestMethod = TestMethod.one_row,
                 full_name=False):
        super().__init__(n_r, n_sc, cp, a=a, use_true_var=use_true_var, testMethod=testMethod, full_name=full_name)

    def name(self):
        return 'sw-test-{}'.format(self.testMethod.name)


class KSTestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        var = true_var if self.use_true_var else est_var
        if self.two_samp:
            _, p = scipy.stats.kstest(idft_row, idft_g[i + 1:].flatten())
        else:
            _, p = scipy.stats.kstest(idft_row, 'norm', (0, var ** 0.5))
        return p <= self.a, p

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        var = true_var if self.use_true_var else est_var
        _, p = scipy.stats.kstest(idft_rows, 'norm', (0, var ** 0.5))
        return p <= self.a, p

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        var = true_var if self.use_true_var else est_var
        _, p = scipy.stats.kstest(dft_diff.flatten(), 'norm', (0, var ** 0.5))
        return p <= self.a, p

    def __init__(self, n_r, n_sc, cp, two_samp=True, a=0.05, use_true_var=False,
                 testMethod: TestMethod = TestMethod.one_row,
                 full_name=False):
        super().__init__(n_r, n_sc, cp, a=a, use_true_var=use_true_var, testMethod=testMethod, full_name=full_name)
        self.two_samp = two_samp

    def name(self):
        n = 'ks-test-{}'.format(self.testMethod.name)
        if self.testMethod == TestMethod.one_row:
            n += '-ts{}'.format(self.two_samp)
        return n


class ADTestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        statistic, critical_values, significance_level = scipy.stats.anderson_ksamp(
            [idft_row, idft_g[i + 1:].flatten()])
        return statistic > list(critical_values)[self.significance_level], statistic

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        statistic, critical_values, significance_level = scipy.stats.anderson(idft_rows, 'norm')
        return statistic > list(critical_values)[self.significance_level], statistic

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        statistic, critical_values, significance_level = scipy.stats.anderson(dft_diff.flatten(), 'norm')
        return statistic > list(critical_values)[self.significance_level], statistic

    def __init__(self, n_r, n_sc, cp, significance_level=4, a=0.05, use_true_var=False,
                 testMethod: TestMethod = TestMethod.one_row,
                 full_name=False):
        super().__init__(n_r, n_sc, cp, a=a, use_true_var=use_true_var, testMethod=testMethod, full_name=full_name)
        self.significance_level = significance_level

    def name(self):
        return 'ad-test-{}'.format(self.testMethod.name)


class NormalTestMethod(DftChuckTestMethod):

    def test_one_row(self, idft_row: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        _, p = scipy.stats.normaltest(idft_row)
        return p <= self.a, p

    def test_whole_noise(self, idft_rows: np.ndarray, est_var, true_var, i, idft_g) -> (bool, any):
        _, p = scipy.stats.normaltest(idft_rows)
        return p <= self.a, p

    def test_dft_diff(self, dft_diff: np.ndarray, est_var, true_var, i, dft_chuck_g) -> (bool, any):
        _, p = scipy.stats.normaltest(dft_diff.flatten())
        return p <= self.a, p

    def __init__(self, n_r, n_sc, cp, a=0.05, use_true_var=False, testMethod: TestMethod = TestMethod.one_row,
                 full_name=False):
        super().__init__(n_r, n_sc, cp, a=a, use_true_var=use_true_var, testMethod=testMethod, full_name=full_name)

    def name(self):
        return 'normal-test-{}'.format(self.testMethod.name)
