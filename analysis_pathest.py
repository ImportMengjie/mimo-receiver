import numpy as np
import scipy.stats

from loader import CsiDataloader, DataType
from model import PathEstDnn, PathEstCnn
from train import load_model_from_file
from utils import DenoisingMethodLS
from utils import VarTestMethod, SWTestMethod, KSTestMethod, ADTestMethod, NormalTestMethod, ModelPathestMethod
from utils import draw_line
from utils.DftChuckTestMethod import TestMethod, Transform, DftChuckThresholdMethod
import config.config as config
import scipy.fftpack as sp

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu

# test var-test diff method
cmp_var_diff_method = lambda n_r, cp, n_sc: [
    VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row),
    # VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.whole_noise),
    VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff)]

# test sw-test diff method
cmp_sw_diff_method = lambda n_r, cp, n_sc: [
    SWTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row),
    # SWTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.whole_noise),
    SWTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff)]

# test ks-test diff method
cmp_ks_diff_method = lambda n_r, cp, n_sc: [
    # KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row, two_samp=True),
    KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row, two_samp=False),
    # KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.whole_noise),
    KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff)]

# test ad-test diff method
cmp_ad_diff_method = lambda n_r, cp, n_sc: [
    ADTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row),
    # ADTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.whole_noise),
    ADTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff)]

# test normal-test diff method
cmp_normal_diff_method = lambda n_r, cp, n_sc: [
    NormalTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row),
    # NormalTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.whole_noise),
    NormalTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff)]


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


def cmp_diff_test_method(csi_loader, dft_chuck_test_list, snr_start, snr_end, snr_step, fix_path=None, draw_nmse=True):
    xp = csi_loader.get_pilot_x()
    h = csi_loader.get_h(DataType.test)
    hx = h @ xp
    g_len = h.shape[0] * h.shape[-1]
    snr_right_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_error_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_total_est_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_error_over_count = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    mse_chuck = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    ls_mse = [0 for _ in range(snr_start, snr_end, snr_step)]
    dft_chuck_mse = [0 for _ in range(snr_start, snr_end, snr_step)]

    get_path_count = lambda idx: fix_path if fix_path is not None else csi_loader.path_count[idx]
    for snr in range(snr_start, snr_end, snr_step):
        logging.info('start snr {}'.format(snr))
        n, var = csi_loader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        h_hat = DenoisingMethodLS().get_h_hat(y, h, xp, var, csi_loader.rhh)
        g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape(-1, csi_loader.n_sc, csi_loader.n_r)
        g = h.permute(0, 3, 1, 2).numpy().reshape(-1, csi_loader.n_sc, csi_loader.n_r)
        g_hat_idft = np.fft.ifft(g_hat, axis=-2)
        g_hat_idct = sp.idct(g_hat, axis=-2, norm='ortho')

        for i in range(g_hat.shape[0]):
            i_g = g[i]
            i_g_hat_idft = g_hat_idft[i]
            i_g_hat_idct = g_hat_idct[i]
            i_g_hat = g_hat[i]
            true_var = var[i // csi_loader.n_t, 0, 0].item() / 2
            path_mse_dict_dft = {}
            path_mse_dict_dct = {}

            def get_path_hat_mse(path_hat: int, t: Transform):
                if t == Transform.dft:
                    if path_hat not in path_mse_dict_dft:
                        chuck_array = np.concatenate((np.ones(path_hat), np.zeros(csi_loader.n_sc - path_hat))).reshape(
                            (-1, 1))
                        i_g_hat_chuck_idft = i_g_hat_idft * chuck_array
                        i_g_hat_chuck = np.fft.fft(i_g_hat_chuck_idft, axis=-2)
                        path_mse_dict_dft[path_hat] = (np.abs(i_g_hat_chuck - i_g) ** 2).sum(-1).sum(-1) / (
                                np.abs(i_g) ** 2).sum(-1).sum(-1)
                    return path_mse_dict_dft[path_hat]
                else:
                    if path_hat not in path_mse_dict_dct:
                        chuck_array = np.concatenate((np.ones(path_hat), np.zeros(csi_loader.n_sc - path_hat))).reshape(
                            (-1, 1))
                        i_g_hat_chuck_idft = i_g_hat_idft * chuck_array
                        i_g_hat_chuck = sp.dct(i_g_hat_chuck_idft, axis=-2, norm='ortho')
                        path_mse_dict_dft[path_hat] = (np.abs(i_g_hat_chuck - i_g) ** 2).sum(-1).sum(-1) / (
                                np.abs(i_g) ** 2).sum(-1).sum(-1)
                    return path_mse_dict_dft[path_hat]

            if draw_nmse:
                ls_mse[(snr - snr_start) // snr_step] += get_path_hat_mse(csi_loader.n_sc, Transform.dft)
                dft_chuck_mse[(snr - snr_start) // snr_step] += get_path_hat_mse(get_path_count(i), Transform.dft)

            for dft_idx in range(len(dft_chuck_test_list)):
                snr_total_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                path_hat, p_list = dft_chuck_test_list[dft_idx].get_path_count(i_g_hat_idft, i_g_hat, true_var)
                if path_hat != get_path_count(i):
                    snr_error_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                    if path_hat > get_path_count(i):
                        snr_error_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                else:
                    snr_right_est_count[dft_idx][(snr - snr_start) // snr_step] += 1
                if draw_nmse:
                    mse_chuck[dft_idx][(snr - snr_start) // snr_step] += get_path_hat_mse(path_hat)

    draw_nmse_dict = {}
    draw_right_dict = {}
    draw_over_error_dict = {}
    for dft_idx in range(len(dft_chuck_test_list)):
        right_est_count = np.array(snr_right_est_count[dft_idx])
        error_est_count = np.array(snr_error_est_count[dft_idx])
        total_est_count = np.array(snr_total_est_count[dft_idx])
        over_est_count = np.array(snr_error_over_count[dft_idx])
        draw_over_error_dict[dft_chuck_test_list[dft_idx].name()] = over_est_count / error_est_count * 100
        draw_right_dict[dft_chuck_test_list[dft_idx].name()] = right_est_count / total_est_count * 100

        if draw_nmse:
            mse_sum = np.array(mse_chuck[dft_idx])
            draw_nmse_dict[dft_chuck_test_list[dft_idx].name()] = 10 * np.log10(mse_sum / g_len)

    draw_line(list(range(snr_start, snr_end, snr_step)), draw_right_dict, '{}-path-est-cmp'.format(csi_loader),
              diff_line_markers=True, ylabel='%', save_dir=config.PATHEST_RESULT_IMG)
    draw_line(list(range(snr_start, snr_end, snr_step)), draw_over_error_dict,
              '{}-over-path-est-cmp'.format(csi_loader), diff_line_markers=True, ylabel='%',
              save_dir=config.PATHEST_RESULT_IMG)
    if draw_nmse:
        draw_nmse_dict['ls'] = 10 * np.log10(np.array(ls_mse) / g_len)
        draw_nmse_dict['dft_chuck'] = 10 * np.log10(np.array(dft_chuck_mse) / g_len)
        draw_line(list(range(snr_start, snr_end, snr_step)), draw_nmse_dict,
                  '{}-path-est-nmse-cmp'.format(csi_loader), diff_line_markers=True, save_dir=config.PATHEST_RESULT_IMG)


def cmp_diff_test_method_nmse(csi_loader, dft_chuck_test_list, snr_start, snr_end, snr_step, fix_path=None, ):
    xp = csi_loader.get_pilot_x()
    h = csi_loader.get_h(DataType.train)
    hx = h @ xp
    h_len = h.shape[0] * csi_loader.n_sc

    snr_h_mse = [[0 for _ in range(snr_start, snr_end, snr_step)] for _ in dft_chuck_test_list]
    snr_h_ls_mse = [0 for _ in range(snr_start, snr_end, snr_step)]
    snr_h_dft_mse = [0 for _ in range(snr_start, snr_end, snr_step)]
    snr_h_dct_mse = [0 for _ in range(snr_start, snr_end, snr_step)]

    get_true_path_count = lambda idx: fix_path if fix_path is not None else csi_loader.path_count[idx]
    for snr in range(snr_start, snr_end, snr_step):
        logging.info("start snr:{}".format(snr))
        n, var = csi_loader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        h_hat = DenoisingMethodLS().get_h_hat(y, h, xp, var, csi_loader.rhh)
        g_hat = h_hat.permute(0, 3, 1, 2).numpy()
        g = h.permute(0, 3, 1, 2).numpy()
        g_hat_idft = np.fft.ifft(g_hat, axis=-2)
        g_hat_idct = sp.idct(g_hat, axis=-2, norm='ortho')
        for j in range(g.shape[0]):
            j_h_chuck = [None for _ in range(len(dft_chuck_test_list) + 3)]
            true_var = var[j, 0, 0].item() / 2
            for m in range(g.shape[1]):
                i_g_hat = g_hat[j][m]
                i_g_hat_idft = g_hat_idft[j][m]
                i_g_hat_idct = g_hat_idct[j][m]
                path_g_dict_dft = {}
                path_g_dict_dct = {}

                def get_chuck_g(path_hat: int, t: Transform):
                    if t == Transform.dft:
                        if path_hat not in path_g_dict_dft:
                            chuck_array = np.concatenate(
                                (np.ones(path_hat), np.zeros(csi_loader.n_sc - path_hat))).reshape(
                                (-1, 1))
                            i_g_hat_chuck_idft = i_g_hat_idft * chuck_array
                            i_g_hat_chuck = np.fft.fft(i_g_hat_chuck_idft, axis=-2, )
                            path_g_dict_dft[path_hat] = i_g_hat_chuck.reshape((1,) + i_g_hat_chuck.shape)
                        return path_g_dict_dft[path_hat]
                    else:
                        if path_hat not in path_g_dict_dct:
                            chuck_array = np.concatenate(
                                (np.ones(path_hat), np.zeros(csi_loader.n_sc - path_hat))).reshape(
                                (-1, 1))
                            i_g_hat_chuck_idct = i_g_hat_idct * chuck_array
                            i_g_hat_chuck = sp.dct(i_g_hat_chuck_idct, axis=-2, norm='ortho')
                            path_g_dict_dct[path_hat] = i_g_hat_chuck.reshape((1,) + i_g_hat_chuck.shape)
                        return path_g_dict_dct[path_hat]

                def put_in_j_h_chuck(path_hat, idx, t: Transform):
                    if j_h_chuck[idx] is None:
                        j_h_chuck[idx] = get_chuck_g(path_hat, t)
                    else:
                        j_h_chuck[idx] = np.concatenate((j_h_chuck[idx], get_chuck_g(path_hat, t)), axis=0)

                # ls
                put_in_j_h_chuck(csi_loader.n_sc, -1, Transform.dft)
                # cp chuck
                put_in_j_h_chuck(cp, -2, Transform.dft)
                put_in_j_h_chuck(cp, -3, Transform.dct)
                for dft_idx in range(len(dft_chuck_test_list)):
                    path_hat, p_list = dft_chuck_test_list[dft_idx].get_path_count(i_g_hat_idft, i_g_hat_idct, i_g_hat,
                                                                                   true_var)
                    put_in_j_h_chuck(path_hat, dft_idx, dft_chuck_test_list[dft_idx].transform)

            true_h = h[j].permute(2, 0, 1).numpy()

            def calc_mse(idx):
                return ((np.abs(j_h_chuck[idx] - true_h) ** 2).sum(0).sum(-1) / (np.abs(true_h) ** 2).sum(0).sum(
                    -1)).sum().item()

            for dft_idx in range(len(dft_chuck_test_list)):
                snr_h_mse[dft_idx][(snr - snr_start) // snr_step] += calc_mse(dft_idx)
            snr_h_ls_mse[(snr - snr_start) // snr_step] += calc_mse(-1)
            snr_h_dft_mse[(snr - snr_start) // snr_step] += calc_mse(-2)
            snr_h_dct_mse[(snr - snr_start) // snr_step] += calc_mse(-3)

    draw_h_nmse_dict = {}
    for dft_idx in range(len(dft_chuck_test_list)):
        mse_sum = np.array(snr_h_mse[dft_idx])
        draw_h_nmse_dict[dft_chuck_test_list[dft_idx].name()] = 10 * np.log10(mse_sum / h_len)

    draw_h_nmse_dict['ls'] = 10 * np.log10(np.array(snr_h_ls_mse) / h_len)
    draw_h_nmse_dict['dft_cp_chuck'] = 10 * np.log10(np.array(snr_h_dft_mse) / h_len)
    draw_h_nmse_dict['dct_cp_chuck'] = 10 * np.log10(np.array(snr_h_dct_mse) / h_len)

    draw_line(list(range(snr_start, snr_end, snr_step)), draw_h_nmse_dict,
              '{}-path-est-h-nmse-cmp'.format(csi_loader), diff_line_markers=True, save_dir=config.PATHEST_RESULT_IMG)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    cp = 64
    min_path = 1
    data_path = 'data/imt_2020_64_32_512_100.mat'
    csi_loader = CsiDataloader(data_path, train_data_radio=0.1)
    n_r = csi_loader.n_r
    n_sc = csi_loader.n_sc

    model_dnn = PathEstDnn(csiDataloader=csi_loader, add_var=True, use_true_var=False,
                           dnn_list=[256, 256, 128, 128, 64, 32],
                           extra='')
    model_dnn = load_model_from_file(model_dnn, use_gpu)
    model_dnn.name = 'dnn'

    model_cnn = PathEstCnn(csiDataloader=csi_loader, add_var=True, use_true_var=False, cnn_count=4, cnn_channel=32,
                           dnn_list=[2000, 200, 20], extra='')
    model_cnn = load_model_from_file(model_cnn, use_gpu)
    model_cnn.name = 'cnn'

    dft_chuck_test_list = []
    # dft_chuck_test_list.append(ModelPathestMethod(n_r=n_r, cp=cp, n_sc=n_sc, model=model_dnn))
    # dft_chuck_test_list.append(ModelPathestMethod(n_r=n_r, cp=cp, n_sc=n_sc, model=model_cnn))

    # threshold
    dft_chuck_test_list.append(
        DftChuckThresholdMethod(n_r=n_r, n_sc=n_sc, cp=cp, min_path=min_path, full_name=False, transform=Transform.dft))
    dft_chuck_test_list.append(
        DftChuckThresholdMethod(n_r=n_r, n_sc=n_sc, cp=cp, min_path=min_path, full_name=False, transform=Transform.dct))

    # var-test
    # dft_chuck_test_list.append(VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row))
    # dft_chuck_test_list.append(VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff))
    # dft_chuck_test_list.append(
    #     VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, transform=Transform.dct, testMethod=TestMethod.one_row))
    dft_chuck_test_list.append(
        VarTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, transform=Transform.dct, testMethod=TestMethod.dft_diff))

    # ks-test
    # dft_chuck_test_list.append(KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row))
    # dft_chuck_test_list.append(KSTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff))

    # ad-test
    # dft_chuck_test_list.append(ADTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row))
    # dft_chuck_test_list.append(ADTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff))

    # sw-test
    # dft_chuck_test_list.append(SWTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row))
    # dft_chuck_test_list.append(SWTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff))

    # norm-test
    # dft_chuck_test_list.append(NormalTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.one_row))
    # dft_chuck_test_list.append(NormalTestMethod(n_r=n_r, cp=cp, n_sc=n_sc, testMethod=TestMethod.dft_diff))

    cmp_diff_test_method_nmse(csi_loader=csi_loader, dft_chuck_test_list=dft_chuck_test_list, snr_start=0, snr_end=15,
                              snr_step=2)
    # cmp_diff_test_method(csi_loader=csi_loader, dft_chuck_test_list=dft_chuck_test_list, snr_start=0, snr_end=5,
    #                      snr_step=1,)

    # analysis_dft_denosing(data_path=data_path, fix_snr=2, max_count=3, path_start=1,
    #                       path_end=20)
