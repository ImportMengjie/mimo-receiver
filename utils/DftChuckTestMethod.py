import numpy as np
import abc

import scipy.stats
from statsmodels.stats.diagnostic import lilliefors


class DftChuckTestMethod(abc.ABC):

    def __init__(self, n_r, cp, a=0.05):
        self.n_r = n_r
        self.cp = cp
        self.a = a
        self.not_cp = n_r - cp

    @abc.abstractmethod
    def get_path_count(self, g_idft: np.ndarray) -> int:
        pass

    @abc.abstractmethod
    def name(self):
        return 'abs_dft_chuck'


class VarTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp):
        super().__init__(n_r, cp)

    def get_path_count(self, g_idft):
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

    def __init__(self, n_r, cp):
        super().__init__(n_r, cp)

    def get_path_count(self, g_idft: np.ndarray):
        for i in range(self.cp - 1, 0, -1):
            noise = g_idft[i:].flatten()
            w, p = scipy.stats.shapiro(noise)
            if p <= self.a:
                return i + 1
        return 0

    def name(self):
        return 'sw-test'


class KSTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp):
        super().__init__(n_r, cp)

    def get_path_count(self, g_idft: np.ndarray):
        for i in range(self.cp - 1, 0, -1):
            noise = g_idft[i:].flatten()
            _, p = lilliefors(noise)
            if p <= self.a:
                return i + 1
        return 0

    def name(self):
        return 'ks-test'


class ADTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp, significance_level=1):
        super().__init__(n_r, cp)
        self.significance_level = significance_level

    def get_path_count(self, g_idft: np.ndarray) -> int:
        for i in range(self.cp - 1, 0, -1):
            noise = g_idft[i:].flatten()
            statistic, critical_values, significance_level = scipy.stats.anderson(noise, 'norm')
            if statistic > list(critical_values)[self.significance_level]:
                return i + 1
        return 0

    def name(self):
        return 'ad-test'


class NormalTestMethod(DftChuckTestMethod):

    def __init__(self, n_r, cp):
        super().__init__(n_r, cp)

    def get_path_count(self, g_idft: np.ndarray) -> int:
        for i in range(self.cp - 1, 0, -1):
            noise = g_idft[i:].flatten()
            _, p = scipy.stats.normaltest(noise)
            if p <= self.a:
                return i + 1
        return 0

    def name(self):
        return 'normal-test'
