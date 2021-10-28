import numpy as np
import abc
import functools

import scipy.stats
from scipy.stats import distributions
from scipy.stats.morestats import AndersonResult
from statsmodels.stats.diagnostic import lilliefors


class DftChuckTestMethod(abc.ABC):

    def __init__(self, n_r, cp, a=0.02, use_true_var=True, test_one_row=True):
        self.n_r = n_r
        self.cp = cp
        self.a = a
        self.not_cp = n_r - cp
        self.test_one_row = test_one_row
        self.use_true_var = use_true_var

    @abc.abstractmethod
    def get_path_count(self, g_idft: np.ndarray, var) -> int:
        pass

    @abc.abstractmethod
    def name(self):
        return 'abs_dft_chuck'


class VarTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp):
        super().__init__(n_r, cp)

    def get_path_count(self, g_idft, var):
        var = np.var(g_idft[self.cp:, :])
        for i in range(self.cp - 1, 0, -1):
            cur_row = g_idft[i]
            cur_row_chi = (len(cur_row) - 1) * np.var(cur_row) / var
            p = 1 - scipy.stats.chi2(len(cur_row) - 1).cdf(cur_row_chi)
            if p <= self.a:
                return i + 1
        return 0

    def name(self):
        return 'var-test'


class SWTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, test_one_row=True):
        super().__init__(n_r, cp, test_one_row=test_one_row)

    def get_path_count(self, g_idft: np.ndarray, var):
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i].flatten()
                w, p = scipy.stats.shapiro(noise)
                if p <= self.a:
                    return i + 1
        else:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i:].flatten()
                w, p = scipy.stats.shapiro(noise)
                if p <= self.a:
                    return i + 1
        return 0

    def name(self):
        return 'sw-test'


class KSTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, test_one_row=True):
        super().__init__(n_r, cp, test_one_row=test_one_row)

    def get_path_count(self, g_idft: np.ndarray, var):
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i].flatten()
                if self.use_true_var:
                    pre_var = var
                else:
                    pre_var = (g_idft[i + 1:] ** 2).mean()
                cdf = functools.partial(scipy.stats.norm.cdf, loc=0, scale=pre_var ** 0.5)
                # _, p = scipy.stats.kstest(noise, 'norm', (0, pre_var ** 0.5))
                _, p = scipy.stats.kstest(noise, g_idft[i+1:].flatten())
                if p <= self.a:
                    return i + 1
        else:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i:].flatten()
                _, p = lilliefors(noise)
                if p <= self.a:
                    return i + 1
        return 0

    def name(self):
        return 'ks-test'


class ADTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, test_one_row=False, significance_level=1):
        super().__init__(n_r, cp, test_one_row=test_one_row)
        self.significance_level = significance_level

    def get_path_count(self, g_idft: np.ndarray, var) -> int:
        _Avals_norm = np.array([0.576, 0.656, 0.787, 0.918, 1.092])
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                pre_noise = g_idft[i + 1:]
                if self.use_true_var:
                    pre_var = var
                else:
                    pre_var = (pre_noise ** 2).sum() / np.size(pre_noise)
                # pre_std = np.std(pre_noise, ddof=1)
                xbar = 0
                x = g_idft[i].flatten()
                y = np.sort(x)
                N = len(y)
                w = (y - xbar) / pre_var ** 0.5
                logcdf = distributions.norm.logcdf(w)
                logsf = distributions.norm.logsf(w)
                sig = np.array([15, 10, 5, 2.5, 1])
                critical = np.around(_Avals_norm / (1.0 + 4.0 / N - 25.0 / N / N), 3)
                idx = np.arange(1, N + 1)
                A2 = -N - np.sum((2 * idx - 1.0) / N * (logcdf + logsf[::-1]), axis=0)

                statistic, critical_values, significance_level = AndersonResult(A2, critical, sig)
                if statistic > list(critical_values)[self.significance_level]:
                    return i + 1
        else:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i:].flatten()
                statistic, critical_values, significance_level = scipy.stats.anderson(noise, 'norm')
                if statistic > list(critical_values)[self.significance_level]:
                    return i + 1
        return 0

    def name(self):
        return 'ad-test'


class NormalTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, test_one_row=True):
        super().__init__(n_r, cp, test_one_row=test_one_row)

    def get_path_count(self, g_idft: np.ndarray, var) -> int:
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i].flatten()
                _, p = scipy.stats.normaltest(noise)
                if p <= self.a:
                    return i + 1
        else:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i:].flatten()
                _, p = scipy.stats.normaltest(noise)
                if p <= self.a:
                    return i + 1
        return 0

    def name(self):
        return 'normal-test'
