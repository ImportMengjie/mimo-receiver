import numpy as np
import scipy.stats

from loader import CsiDataloader, DataType
from model import DnnPathEst
from train import load_model_from_file
from utils import DenoisingMethodLS
from utils import VarTestMethod, SWTestMethod, KSTestMethod, ADTestMethod, NormalTestMethod, DnnModelPathestMethod
from utils import draw_line
from utils.DftChuckTestMethod import TestMethod
import utils.config as config

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu


def flatten_complex(a):
    a = a.reshape(a.shape + (1,))
    a = np.concatenate((np.real(a), np.imag(a)), axis=-1).flatten()
    return a


def analysis_dft_denosing(data_path, fix_snr, max_count, path_start, path_end=None, cp=None):
    csi_loader = CsiDataloader(data_path, train_data_radio=1)
    if path_end is None:
        path_end = csi_loader.n_sc + 1
    if cp is None:
        cp = path_end
    xp = csi_loader.get_pilot_x()
    h = csi_loader.get_h(DataType.train)
    hx = h @ xp
    n, var = csi_loader.noise_snr_range(hx, [fix_snr, fix_snr + 1], one_col=False)
    y = hx + n
    h_hat = DenoisingMethodLS().get_h_hat(y, h, xp, var, csi_loader.rhh)
    g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape((-1, csi_loader.n_sc, csi_loader.n_r))
    g = h.permute(0, 3, 1, 2).numpy().reshape((-1, csi_loader.n_sc, csi_loader.n_r))

    g_hat = g_hat[0:max_count]
    g = g[0:max_count]

    g_hat_idft = np.fft.ifft(g_hat, axis=-2)
    g_idft = np.fft.ifft(g, axis=-2)
    nmse_dict = {str(i): [] for i in range(g.shape[0])}
    nmse_dict.update({str(i) + 'ls': [] for i in range(g.shape[0])})
    true_nmse_dict = {str(i): [] for i in range(g.shape[0])}
    noise_var = {str(i): [] for i in range(g.shape[0])}
    noise_mean = {str(i): [] for i in range(g.shape[0])}
    noise_up = {str(i): [] for i in range(g.shape[0])}
    chuck_noise = {str(i): [] for i in range(g.shape[0])}
    p_value = {str(i): [] for i in range(g.shape[0])}

    chuck_noise_fix = {str(i) + '-chuck_noise_var_fix': [] for i in range(g.shape[0])}

    def get_path_data(path_count):
        chuck_array = np.concatenate((np.ones(path_count), np.zeros(csi_loader.n_sc - path_count))).reshape((-1, 1))
        not_chuck_array = np.concatenate(
            (np.full((path_count,), False), np.full((csi_loader.n_sc - path_count,), True)))
        g_hat_idft_chuck = g_hat_idft * chuck_array
        g_idft_chuck = g_idft * chuck_array
        g_hat_idft_noise = g_hat_idft[:, not_chuck_array, :]
        g_hat_chuck = np.fft.fft(g_hat_idft_chuck, axis=-2)
        g_chuck = np.fft.fft(g_idft_chuck, axis=-2)
        return chuck_array, not_chuck_array, g_hat_idft_chuck, g_idft_chuck, g_hat_idft_noise, g_hat_chuck, g_chuck

    _, _, _, _, cp_hat_idft_noise, _, _ = get_path_data(cp)

    for path_count in range(path_start, path_end):
        chuck_array, not_chuck_array, g_hat_idft_chuck, g_idft_chuck, g_hat_idft_noise, g_hat_chuck, g_chuck = get_path_data(
            path_count)
        for i in range(g_hat_chuck.shape[0]):
            i_g_hat_chuck = g_hat_chuck[i]
            i_g = g[i]
            i_g_chuck = g_chuck[i]
            i_g_hat = g_hat[i]
            i_g_idft_noise = g_hat_idft_noise[i]
            i_nmse = 10 * np.log10(((np.abs(i_g - i_g_hat_chuck) ** 2).sum() / (np.abs(i_g) ** 2).sum()).mean())
            nmse_dict[str(i)].append(i_nmse)
            ls_nmse = 10 * np.log10(((np.abs(i_g - i_g_hat) ** 2).sum() / (np.abs(i_g) ** 2).sum()).mean())
            nmse_dict[str(i) + 'ls'].append(ls_nmse)

            i_true_nmse = 10 * np.log10(((np.abs(i_g - i_g_chuck) ** 2).sum() / (np.abs(i_g) ** 2).sum()).mean())
            true_nmse_dict[str(i)].append(i_true_nmse)

            i_n = (i_g_hat_chuck - i_g_hat)
            i_n = i_n.reshape(i_n.shape + (1,))
            i_n = np.concatenate((np.real(i_n), np.imag(i_n)), axis=-1).flatten()
            # noise_mean[str(i)].append(i_n.mean())
            noise_var[str(i)].append(i_n.var())
            # noise_up[str(i)].append((i_n > i_n.mean()).sum() / len(i_n))

            i_g_idft_noise = i_g_idft_noise.reshape(i_g_idft_noise.shape + (1,))
            i_g_idft_noise = np.concatenate((np.real(i_g_idft_noise), np.imag(i_g_idft_noise)), axis=-1).flatten()
            chuck_noise[str(i)].append(i_g_idft_noise.var())
            noise_mean[str(i)].append(i_g_idft_noise.mean())
            noise_up[str(i)].append((i_g_idft_noise > i_g_idft_noise.mean()).sum() / len(i_n))
            chuck_noise_fix[str(i) + '-chuck_noise_var_fix'].append(
                i_g_idft_noise.var() * (csi_loader.n_sc - path_count))

            i_cp_hat_idft_noise = cp_hat_idft_noise[i].reshape(cp_hat_idft_noise[i].shape + (1,))
            i_cp_hat_idft_noise = np.concatenate((np.real(i_cp_hat_idft_noise), np.imag(i_cp_hat_idft_noise)),
                                                 axis=-1).flatten()
            i_cp_var = i_cp_hat_idft_noise.var()
            cur_hat_idft_noise = flatten_complex(g_hat_idft[i, path_count - 1])
            cur_path_chi = (csi_loader.n_r - 1) * np.var(cur_hat_idft_noise) / i_cp_var
            cur_path_p = 1 - scipy.stats.chi2(csi_loader.n_r - 1).cdf(cur_path_chi)
            p_value[str(i)].append(cur_path_p)

    chuck_noise_fix.update(noise_var)
    x = list(range(path_start, path_end))
    noise_var['var'] = [var.mean().item() / 2 for _ in x]
    # chuck_noise['var'] = [var.mean().item() / 2 for _ in x]
    min_path_txt = 'min_path'
    for k, v in nmse_dict.items():
        if k.isdigit():
            min_path = np.array(v).argmin() + path_start
            min_path_txt += '{}:{};'.format(k, min_path)

    title = '{}'.format(csi_loader)
    draw_line(x, nmse_dict, title="{}-{}db".format(title, fix_snr) + min_path_txt, xlabel='chuck_path',
              diff_line_markers=False)
    # draw_line(x, true_nmse_dict, title="true{}db".format(fix_snr) + min_path_txt, xlabel='chuck_path',
    #           diff_line_markers=True)
    # draw_line(x, noise_mean, xlabel='chuck_path', ylabel='mean', diff_line_markers=True)
    draw_line(x, noise_var, title=title, xlabel='chuck_path', ylabel='var', diff_line_markers=True)
    # draw_line(x, noise_up, xlabel='chuck_path', ylabel='up/total', diff_line_markers=True)
    draw_line(x, chuck_noise, xlabel='chuck_path', ylabel='var', title=title + '-chuck-var', diff_line_markers=True)
    draw_line(x, chuck_noise_fix, xlabel='chuck_path', ylabel='var', title='compare', diff_line_markers=True)
    draw_line(x, p_value, xlabel='chuck_path', ylabel='p-value', title='{}-p_value'.format(title),
              diff_line_markers=True)


def cmp_diff_test_method(data_path, snr_start, snr_end, snr_step, fix_path=None):
    csi_loader = CsiDataloader(data_path, train_data_radio=1)
    xp = csi_loader.get_pilot_x()
    h = csi_loader.get_h(DataType.train)[:1000 // csi_loader.n_t]
    hx = h @ xp
    cp = 20

    model = DnnPathEst(csiDataloader=csi_loader, add_var=True, use_true_var=False,
                       dnn_list=[256, 256, 128, 128, 64, 32],
                       extra='')
    model = load_model_from_file(model, use_gpu)
    # dft_chuck_test_list = [VarTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc),
    #                        SWTestMethod(csi_loader.n_r, 20),
    #                        KSTestMethod(csi_loader.n_r, 20),
    #                        ADTestMethod(csi_loader.n_r, 20, significance_level=4),
    #                        NormalTestMethod(csi_loader.n_r, 20)]

    # test var-test diff method
    # dft_chuck_test_list = [
    #     VarTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row),
    #     VarTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.whole_noise),
    #     VarTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.dft_diff)]

    # test sw-test diff method
    # dft_chuck_test_list = [
    #     SWTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row),
    #     SWTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.whole_noise),
    #     SWTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.dft_diff)]

    # test ks-test diff method
    # dft_chuck_test_list = [
    #     KSTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row, two_samp=True),
    #     KSTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row, two_samp=False),
    #     KSTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.whole_noise),
    #     KSTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.dft_diff)]

    # test ad-test diff method
    # dft_chuck_test_list = [
    #     ADTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row),
    #     ADTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.whole_noise),
    #     ADTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.dft_diff)]

    # test normal-test diff method
    # dft_chuck_test_list = [
    #     NormalTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.one_row),
    #     NormalTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.whole_noise),
    #     NormalTestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, testMethod=TestMethod.dft_diff)]

    # test ad-test diff significance level
    # dft_chuck_test_list = [VarTestMethod(csi_loader.n_r, 20),
    #                        ADTestMethod(csi_loader.n_r, 20, significance_level=4, full_name=True),
    #                        ADTestMethod(csi_loader.n_r, 20, significance_level=1, full_name=True),
    #                        ]

    # test model
    dft_chuck_test_list = [DnnModelPathestMethod(n_r=csi_loader.n_r, cp=cp, n_sc=csi_loader.n_sc, model=model,
                                                 testMethod=TestMethod.one_row)]

    snr_right_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_error_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_total_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_error_over_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]

    get_path_count = lambda idx: fix_path if fix_path is not None else csi_loader.path_count[idx]
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_loader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        h_hat = DenoisingMethodLS().get_h_hat(y, h, xp, var, csi_loader.rhh)
        g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape(-1, csi_loader.n_sc, csi_loader.n_r)
        g_hat_idft = np.fft.ifft(g_hat, axis=-2)
        for i in range(g_hat.shape[0]):
            i_g_hat_idft = g_hat_idft[i]
            i_g_hat = g_hat[i]
            true_var = var[i // csi_loader.n_t, 0, 0].item() / 2
            for dft_idx in range(len(dft_chuck_test_list)):
                snr_total_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                path_hat, p_list = dft_chuck_test_list[dft_idx].get_path_count(i_g_hat_idft, i_g_hat, true_var)
                if path_hat != get_path_count(i):
                    snr_error_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                    if path_hat > get_path_count(i):
                        snr_error_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                else:
                    snr_right_est_count[dft_idx][(snr - snr_start) // snr_step] += 1

    draw_right_dict = {}
    draw_over_error_dict = {}
    for dft_idx in range(len(dft_chuck_test_list)):
        right_est_count = np.array(snr_right_est_count[dft_idx])
        error_est_count = np.array(snr_error_est_count[dft_idx])
        total_est_count = np.array(snr_total_est_count[dft_idx])
        over_est_count = np.array(snr_error_over_count[dft_idx])
        draw_over_error_dict[dft_chuck_test_list[dft_idx].name()] = over_est_count / error_est_count * 100
        draw_right_dict[dft_chuck_test_list[dft_idx].name()] = right_est_count / total_est_count * 100
    draw_line(list(range(snr_start, snr_end, snr_step)), draw_right_dict, '{}-path-est-cmp'.format(csi_loader),
              diff_line_markers=True, ylabel='%')
    draw_line(list(range(snr_start, snr_end, snr_step)), draw_over_error_dict,
              '{}-over-path-est-cmp'.format(csi_loader), diff_line_markers=True, ylabel='%')


if __name__ == '__main__':
    # analysis_dft_denosing(data_path="data/spatial_mu_ULA_32_16_64_10_l10_11.mat", fix_snr=2, max_count=3, path_start=1,
    #                       path_end=20)
    cmp_diff_test_method(data_path='data/imt_2020_64_32_64_400.mat', snr_start=0, snr_end=15, snr_step=1)
    # cmp_diff_test_method(data_path='data/spatial_mu_ULA_64_32_64_100_l10_11.mat', snr_start=0, snr_end=15, snr_step=1,
    #                      fix_path=10)
