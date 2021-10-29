import numpy as np
import abc
import functools

import scipy.stats
from scipy.stats import distributions
from scipy.stats.morestats import AndersonResult
from statsmodels.stats.diagnostic import lilliefors


class DftChuckTestMethod(abc.ABC):

    def __init__(self, n_r, cp, a=0.05, use_true_var=True, test_one_row=True, full_name=False):
        self.n_r = n_r
        self.cp = cp
        self.a = a
        self.not_cp = n_r - cp
        self.test_one_row = test_one_row
        self.use_true_var = use_true_var
        self.full_name = full_name

    @abc.abstractmethod
    def get_path_count(self, g_idft: np.ndarray, var) -> int:
        pass

    @abc.abstractmethod
    def name(self):
        return 'abs_dft_chuck'


class VarTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, full_name=False):
        super().__init__(n_r, cp, full_name=full_name)

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

    def __init__(self, n_r, cp, test_one_row=True, two_samp=True, full_name=False):
        super().__init__(n_r, cp, test_one_row=test_one_row, full_name=full_name)
        self.two_samp = two_samp

    def get_path_count(self, g_idft: np.ndarray, var):
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                noise = g_idft[i].flatten()
                if self.use_true_var:
                    pre_var = var
                else:
                    pre_var = (g_idft[i + 1:] ** 2).mean()
                # cdf = functools.partial(scipy.stats.norm.cdf, loc=0, scale=pre_var ** 0.5)
                if self.two_samp:
                    _, p = scipy.stats.kstest(noise, g_idft[i + 1:].flatten())
                else:
                    _, p = scipy.stats.kstest(noise, 'norm', (0, pre_var ** 0.5))
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
        if self.full_name:
            return 'ks-test-or{}-2s{}'.format(self.test_one_row, self.two_samp)
        else:
            return 'ks-test'


class ADTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, test_one_row=True, significance_level=1, two_samp=True, full_name=False):
        super().__init__(n_r, cp, test_one_row=test_one_row, full_name=full_name)
        self.significance_level = significance_level
        self.two_samp = two_samp

    def get_path_count(self, g_idft: np.ndarray, var) -> int:
        _Avals_norm = np.array([0.576, 0.656, 0.787, 0.918, 1.092])
        if self.test_one_row:
            for i in range(self.cp - 1, 0, -1):
                pre_noise = g_idft[i + 1:].flatten()
                if self.two_samp:
                    statistic, critical_values, significance_level = scipy.stats.anderson_ksamp([pre_noise, g_idft[i]])
                    if statistic > list(critical_values)[self.significance_level]:
                        return i + 1
                else:
                    xbar = 0
                    x = g_idft[i].flatten()
                    y = np.sort(x)
                    N = len(y)
                    if self.use_true_var:
                        pre_var = var * (N / (N - 1))
                    else:
                        pre_var = (pre_noise ** 2).sum() / np.size(pre_noise) * (N / (N - 1))
                    # pre_std = np.std(pre_noise, ddof=1)
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
        if self.full_name:
            return 'ad-test-or{}-2s{}-s{}'.format(self.test_one_row, self.two_samp, self.significance_level)
        else:
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
