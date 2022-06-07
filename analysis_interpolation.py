import os
from typing import List

import PIL.Image
import numpy as np
import torch
import scipy.fftpack as sp

import config.config as config
from loader import CsiDataloader, DataType
from model import CBDNetSFModel
from train import load_model_from_file
from utils import Transform
from utils.DenoisingMethod import DenoisingMethodMMSE, DenoisingMethodIdealMMSE, DenoisingMethodLS
from utils.InterpolationMethod import InterpolationMethod, InterpolationMethodLine, InterpolationMethodModel, \
    InterpolationMethodChuck, InterpolationMethodDct, InterpolationMethodTransformChuck, \
    get_transformChuckMethod_fix_path, get_transformChuckMethod_threshold, get_transformChuckMethod_ks, \
    get_modelMethod_fix_path
from utils.draw import draw_line, draw_point_and_line
from utils.common import line_interpolation_hp_pilot, complex2real, get_interpolation_idx_nf
from utils.DftChuckTestMethod import *

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu

denosing_method_list = [DenoisingMethodMMSE(), DenoisingMethodIdealMMSE()]


def analysis_window(csi_dataloader: CsiDataloader):
    def draw(H: np.ndarray, name: str, j, n_r=None):
        """
        :param H: (J,N_t,K,N_r)
        :param name:
        :return:
        """
        h = np.fft.ifft(H, axis=-2)
        per_pic_num = 1
        for i in range(0, N_t, per_pic_num):
            draw_dict = {'{}-{}-{}'.format(j, m, name): None for m in range(i, min(i + per_pic_num, N_t))}
            for m in range(i, min(i + per_pic_num, N_t)):
                if n_r is not None:
                    power = np.abs(h[j, m, :, n_r]) ** 2
                    draw_dict['{}-{}-{}'.format(j, m, name)] = power
                else:
                    power = np.abs(h[j, m, :, :]) ** 2
            draw_line(x=list(range(0, K)), y_dict=draw_dict, xlabel='K', ylabel='power', title=name,
                      diff_line_markers=True)

    def draw_one_h(h_list: List[np.ndarray], name_list: List[str], n_r=None, path_count=None, mold=True):
        draw_dict = {}
        for i in range(len(h_list)):
            if n_r is not None:
                draw_dict[name_list[i]] = abs(h_list[i][:, n_r])
                if not mold:
                    draw_dict[name_list[i]] = draw_dict[name_list[i]] ** 2
            else:
                pass
        ylabel = 'Amplitude' if mold else 'power'
        draw_line(x=list(range(0, h_list[0].shape[0])), y_dict=draw_dict, xlabel='K', ylabel=ylabel,
                  title='', save_dir=config.INTERPOLATION_RESULT_IMG)

    H = csi_dataloader.get_h(DataType.test)

    H = H.permute(0, 3, 1, 2)
    H = H.numpy()
    J, N_t, K, N_r = H.shape
    ifft_h = np.fft.ifft(H, axis=-2)
    idct_h = sp.idct(H, axis=-2, norm='ortho')
    j = 0
    m = 11
    n_r = 0
    draw_one_h([ifft_h[j, m], idct_h[j, m]], ['ifft', 'idct'], n_r=n_r,
               path_count=csi_dataloader.get_path_count(DataType.test, j, m))
    draw_one_h([ifft_h[j, m]], ['ifft', ], n_r=n_r,
               path_count=csi_dataloader.get_path_count(DataType.test, j, m))
    draw_one_h([idct_h[j, m]], ['idct'], n_r=n_r,
               path_count=csi_dataloader.get_path_count(DataType.test, j, m))

    # padding_h = []
    # padding_name = []
    # # padding
    # for n_f in range(1, 4):
    #     idx = get_interpolation_idx_nf(K, n_f).numpy()
    #     H_p = H[j, m, idx]
    #     H_p_idft = np.fft.ifft(H_p, axis=0)
    #     H_p_idft = np.concatenate((H_p_idft, np.zeros((K - H_p.shape[0], H_p.shape[1]))), axis=0)
    #     H_p_idct = sp.idct(H_p, axis=0, norm='ortho')
    #     H_p_idct = np.concatenate((H_p_idct, np.zeros((K - H_p.shape[0], H_p.shape[1]))), axis=0)
    #     padding_h.extend([H_p_idft, H_p_idct])
    #     padding_name.extend(['idft-{}'.format(n_f), 'idct-{}'.format(n_f)])
    # draw_one_h(padding_h, name_list=padding_name, n_r=n_r)


def analysis_interpolation_pilot_data(csi_dataloader: CsiDataloader,
                                      interpolation_method_list: List[InterpolationMethod],
                                      snr_start, snr_end, snr_step):
    pilot_nmse_list = [[] for _ in range(len(interpolation_method_list))]
    data_nmse_list = [[] for _ in range(len(interpolation_method_list))]
    xp = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ xp
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        for i in range(len(interpolation_method_list)):
            pilot_nmse, data_nmse = interpolation_method_list[i].get_pilot_nmse_and_interp_nmse(y, h, xp, var,
                                                                                                csi_dataloader.rhh)
            pilot_nmse_list[i].append(pilot_nmse)
            data_nmse_list[i].append(data_nmse)
    pilot_nmse_k_v = {}
    data_nmse_k_v = {}
    for i in range(len(pilot_nmse_list)):
        if not interpolation_method_list[i].only_est_data:
            pilot_nmse_k_v[interpolation_method_list[i].get_pilot_name()] = pilot_nmse_list[i]
        data_nmse_k_v[interpolation_method_list[i].get_key_name()] = data_nmse_list[i]
    return pilot_nmse_k_v, data_nmse_k_v, list(range(snr_start, snr_end, snr_step))


def analysis_interpolation_total(csi_dataloader: CsiDataloader, interpolation_method_list: List[InterpolationMethod],
                                 snr_start, snr_end, snr_step):
    total_nmse_dict = {method.get_key_name(): [] for method in interpolation_method_list}
    xp = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ xp
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        for i in range(len(interpolation_method_list)):
            total_nmse = interpolation_method_list[i].get_nmse(y, h, xp, var, csi_dataloader.rhh)
            total_nmse_dict[interpolation_method_list[i].get_key_name()].append(total_nmse)
    return total_nmse_dict, list(range(snr_start, snr_end, snr_step))


def draw_pilot_and_data_nmse(csi_dataloader: CsiDataloader, interpolation_method_list: List[InterpolationMethod],
                             snr_start, snr_end, snr_step):
    n_sc, pilot_count, is_denosing = interpolation_method_list[0].n_sc, interpolation_method_list[0].pilot_count, \
                                     interpolation_method_list[0].is_denosing
    pilot_nmse_dict, data_nmse_dict, x = analysis_interpolation_pilot_data(csi_dataloader, interpolation_method_list,
                                                                           snr_start,
                                                                           snr_end, snr_step)
    if is_denosing:
        title = 'block-denosing{}-{}'.format(n_sc, csi_dataloader.__str__())
        draw_line(x, pilot_nmse_dict,
                  title=title,
                  save_dir=config.INTERPOLATION_RESULT_IMG)
    else:
        title = 'comb-interpolation-pilot{}|{}-{}'.format(pilot_count, n_sc, csi_dataloader.__str__())
        draw_line(x, data_nmse_dict,
                  title='comb-interpolation-data{}|{}-{}'.format(n_sc - pilot_count, n_sc, csi_dataloader.__str__()),
                  save_dir=config.INTERPOLATION_RESULT_IMG)


def analysis_h_visualization(csi_dataloader: CsiDataloader, snr, ):
    def draw_g(save_path: str, h: torch.Tensor, ):
        save_path += '_{}.png'
        h_rgb = complex2real(h).detach().numpy()
        h_rgb = 128 / np.ceil(np.abs(h_rgb).max()) * h_rgb + 128
        h_rgb = np.concatenate((h_rgb, np.zeros((h_rgb.shape[0], h_rgb.shape[1], 1))), axis=-1)
        # img = PIL.Image.fromarray(np.uint8(h_rgb), mode='RGB')
        # img.show()
        # with open(save_path.format('rgb'), 'wb') as f:
        #     img.save(f)

        h_grey = h.detach().numpy()
        h_grey = np.concatenate((np.real(h_grey), np.imag(h_grey)), axis=1)
        factor = 128 / np.ceil(np.abs(h_grey).max())
        h_grey = h_grey * factor + 128
        h_grey = np.round(h_grey)
        h_grey = 255 - h_grey
        img = PIL.Image.fromarray(np.uint8(h_grey), mode='L')
        # img.show()
        with open(save_path.format('grey'), 'wb') as f:
            img.save(f)

    cp = csi_dataloader.n_sc // 4

    ks_chuck = get_transformChuckMethod_ks(csi_dataloader, Transform.dct, n_f=0, cp=cp)
    fix_path_chuck = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dct, fix_path=cp, n_f=0, cp=cp)
    model = CBDNetSFModel(csiDataloader=csi_dataloader, chuck_name=fix_path_chuck.get_key_name(), add_var=True, n_f=0, )
    model = load_model_from_file(model, use_gpu)
    model = model.double()
    xp = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    J, n_sc, n_r, n_t = h.shape
    hx = h @ xp
    n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
    y = hx + n
    h_p = h
    h_p = DenoisingMethodLS().get_h_hat(y, h_p, xp, var, csi_dataloader.rhh)
    # h_hat = line_interpolation_hp_pilot(h_p, model.pilot_idx, n_sc)
    j = 0
    n_t = 3
    h_hat = h_p.permute(0, 3, 1, 2)
    save_path = lambda x: os.path.join(config.INTERPOLATION_RESULT_IMG, '{}_{}'.format(x, model.name))
    g_true = h[j, :, :, n_t]
    draw_g(save_path('g_true'), g_true)
    g_hat = h_hat[j, n_t]
    draw_g(save_path('g_hat'), g_hat)
    g_idct = sp.idct(g_hat.numpy(), axis=-2, norm='ortho')
    draw_g(save_path('g_idct'), torch.from_numpy(g_idct))
    path, est_var = ks_chuck.chuckMethod.get_path_count(g_idct, g_idct, g_hat, var[0].squeeze())
    chuck_array = np.concatenate((np.ones(path), np.zeros(n_sc - path))).reshape((-1, 1))
    g_idct_chuck = g_idct * chuck_array
    draw_g(save_path('g_idct_chuck'), torch.from_numpy(g_idct_chuck))
    g_in = sp.dct(g_idct_chuck, axis=-2, norm='ortho')
    draw_g(save_path('g_in'), torch.from_numpy(g_in))
    g_in = complex2real(torch.from_numpy(g_in))
    g_out, = model(g_in.reshape((1,)+g_in.shape), torch.tensor(est_var).reshape(1,1))
    g_out = g_out.squeeze()
    g_out = g_out[:, :, 0] + g_out[:, :, 1] * 1j

    draw_g(save_path('g_out'), g_out)

    #
    # g_in = g_in.reshape((-1,) + g_in.shape)
    # var = var[0].squeeze().reshape((1, 1))
    # g_out, _ = model(complex2real(g_in), var)
    # g_out = g_out.squeeze()
    # draw_g(save_path('g_out'), g_out, model)


def cmp_model_and_base_method(interpolation_methods, csi_dataloader: CsiDataloader, snr_start, snr_end, snr_step):
    draw_pilot_and_data_nmse(csi_dataloader, interpolation_methods, snr_start=snr_start, snr_end=snr_end,
                             snr_step=snr_step)


def cmp_model_block(csi_dataloader: CsiDataloader, snr_start, snr_end, snr_step, ):
    cp = csi_dataloader.n_sc // 4
    interpolation_methods_sp = [
        InterpolationMethodLine(csi_dataloader.n_sc, 0, 'linear', DenoisingMethodLS()),
        get_transformChuckMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=10, n_f=0, cp=cp),
        InterpolationMethodLine(csi_dataloader.n_sc, 0, 'linear', DenoisingMethodMMSE()),
        get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=10, add_var=True, n_f=0, conv=6,
                                 channel=64, kernel_size=(3, 3), cp=cp, extra=''),
        # get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=cp, add_var=False, n_f=0, conv=6,
        #                          channel=64, kernel_size=(3, 3), cp=cp, extra=''),
        # get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=64, add_var=True, n_f=0, conv=6,
        #                          channel=64, kernel_size=(3, 3), cp=cp, extra=''),
        # get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=cp, add_var=True, n_f=0, conv=6,
        #                          channel=64, kernel_size=(3, 3), cp=cp, extra=''),
    ]
    interpolation_methods_imt = [
        InterpolationMethodLine(csi_dataloader.n_sc, 0, 'linear', DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, 0, 'linear', DenoisingMethodMMSE()),
        # InterpolationMethodTransformChuck(csi_dataloader.n_sc, 0, Transform.dft, denoisingMethod=DenoisingMethodLS(),
        #                                   chuckMethod=KSTestMethod(csi_dataloader.n_r, csi_dataloader.n_sc, 0,
        #                                                            testMethod=TestMethod.freq_diff), ),
        # get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=cp, n_f=0, cp=cp),
        # get_modelMethod_fix_path(csi_dataloader, transform=Transform.dct, fix_path=cp, add_var=True, n_f=0, conv=6,
        #                          channel=64, kernel_size=(3, 3), cp=cp, extra=''),
    ]
    interpolation_methods = interpolation_methods_imt
    cmp_model_and_base_method(interpolation_methods, csi_dataloader, snr_start, snr_end, snr_step)


def cmp_model_comb(csi_dataloader: CsiDataloader, n_f, snr_start, snr_end, snr_step, ):
    cp = csi_dataloader.n_sc // 4
    interpolation_methods_sp = [
        InterpolationMethodLine(csi_dataloader.n_sc, n_f, 'linear', DenoisingMethodLS()),
        get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=10, n_f=n_f, cp=cp),
        get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=10, add_var=True, n_f=0, conv=6,
                                 channel=64, kernel_size=(3, 3), cp=cp)
    ]
    interpolation_methods_imt = [
        InterpolationMethodLine(csi_dataloader.n_sc, n_f, 'linear', DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, n_f, 'quadratic', DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, n_f, 'cubic', DenoisingMethodLS()),
        InterpolationMethodTransformChuck(csi_dataloader.n_sc, n_f, Transform.dct, denoisingMethod=DenoisingMethodLS(),
                                          chuckMethod=KSTestMethod(csi_dataloader.n_r, csi_dataloader.n_sc, 0,
                                                                   testMethod=TestMethod.freq_diff), ),
        InterpolationMethodTransformChuck(csi_dataloader.n_sc, n_f, Transform.dct, denoisingMethod=DenoisingMethodLS(),
                                          chuckMethod=DftChuckThresholdMeanMethod(csi_dataloader.n_r,
                                                                                  csi_dataloader.n_sc, 0, )),
    ]
    interpolation_methods = interpolation_methods_sp
    draw_pilot_and_data_nmse(csi_dataloader, interpolation_methods, snr_start=snr_start, snr_end=snr_end,
                             snr_step=snr_step)


def cmp_diff_pilot_count(csi_dataloader: CsiDataloader, pilot_count_list, snr_start, snr_end, snr_step,
                         model_pilot_count, noise_level_conv, noise_channel, noise_dnn, denoising_conv,
                         denoising_channel, kernel_size, use_two_dim, use_true_sigma, only_return_noise_level, extra='',
                         show_name=None, dft_chuck=0, use_dft_padding=False):
    model = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                          noise_channel=noise_channel, noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                          denoising_channel=denoising_channel, kernel_size=kernel_size, use_two_dim=use_two_dim,
                          use_true_sigma=use_true_sigma, only_return_noise_level=only_return_noise_level, extra=extra,
                          dft_chuck=dft_chuck, use_dft_padding=use_dft_padding)

    model = load_model_from_file(model, use_gpu)
    if show_name:
        model.name = show_name
    interpolation_methods = []
    for pilot_count in pilot_count_list:
        model.pilot_count = pilot_count
        for denosing_method in denosing_method_list:
            interpolation_method = InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, denosing_method, False)
            interpolation_method.extra = '-{}/{}'.format(interpolation_method.pilot_count, csi_dataloader.n_sc)
            interpolation_methods.append(interpolation_method)
        interpolation_method = InterpolationMethodModel(model, use_gpu, pilot_count)
        interpolation_method.extra = '-{}/{}'.format(interpolation_method.pilot_count, csi_dataloader.n_sc)
        if interpolation_method.pilot_count == model.pilot_count:
            interpolation_method.extra += '-same train'
        interpolation_methods.append(interpolation_method)
    total_nmse, snr_x = analysis_interpolation_total(csi_dataloader, interpolation_methods, snr_start, snr_end,
                                                     snr_step, )
    draw_line(snr_x, total_nmse, title='cmp diff pilot count', save_dir=config.INTERPOLATION_RESULT_IMG)


def cmp_diff_path_count(data_path_prefix, path_list, perfect_path, snr_start, snr_end, snr_step, ):
    model = None
    snr_x = None
    total_nmse_dict = {}
    csi_dataloader = CsiDataloader(data_path_prefix + '_l{}_{}.mat'.format(perfect_path, perfect_path + 1),
                                   train_data_radio=0)

    interpolation_method = get_modelMethod_fix_path(csi_dataloader, transform=Transform.dft, fix_path=10, add_var=True,
                                                    n_f=0, conv=6,
                                                    channel=64, kernel_size=(3, 3), cp=16, extra='')
    interpolation_method.model.name = 'CBDNet-SF'
    for path in path_list:
        data_path = data_path_prefix + '_l{}_{}.mat'.format(path, path + 1)
        csi_dataloader = CsiDataloader(data_path, train_data_radio=0)
        interpolation_method.chuckMethod.fix_path = path
        interpolation_method.chuckMethod.chuckMethod.fix_path = path
        interpolation_method.extra = '-{}p'.format(path)
        if path == perfect_path:
            interpolation_method.extra += '-same train'
        nmse_dict, snr_x = analysis_interpolation_total(csi_dataloader, [interpolation_method], snr_start, snr_end,
                                                        snr_step)
        total_nmse_dict.update(nmse_dict)

    draw_line(snr_x, total_nmse_dict, title='cmp diff path count', save_dir=config.INTERPOLATION_RESULT_IMG)


def cmp_model_use_2dim(csi_dataloader: CsiDataloader, pilot_count, snr_start, snr_end, snr_step,
                       model_pilot_count, noise_level_conv, noise_channel, noise_dnn, denoising_conv,
                       denoising_channel, kernel_size, use_true_sigma, only_return_noise_level,
                       extra='', show_name=None):
    model_with_2dim = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                                    noise_channel=noise_channel,
                                    noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                                    denoising_channel=denoising_channel,
                                    kernel_size=kernel_size, use_two_dim=True, use_true_sigma=use_true_sigma,
                                    only_return_noise_level=only_return_noise_level, extra=extra)
    model_with_2dim = load_model_from_file(model_with_2dim, use_gpu)

    model_without_2dim = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                                       noise_channel=noise_channel,
                                       noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                                       denoising_channel=denoising_channel,
                                       kernel_size=kernel_size, use_two_dim=False, use_true_sigma=use_true_sigma,
                                       only_return_noise_level=only_return_noise_level, extra=extra)
    model_without_2dim = load_model_from_file(model_without_2dim, use_gpu)
    if show_name:
        model_with_2dim.name = show_name + "-with2dim"
        model_without_2dim.name = show_name + "-without2dim"
    interpolation_methods = [
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE()),
        InterpolationMethodModel(model_with_2dim, use_gpu, pilot_count),
        InterpolationMethodModel(model_without_2dim, use_gpu, pilot_count)
    ]
    draw_pilot_and_data_nmse(csi_dataloader, interpolation_methods, snr_start=snr_start, snr_end=snr_end,
                             snr_step=snr_step)


def analysis_noise_level(csi_dataloader: CsiDataloader, pilot_count, snr_list, noise_level_conv, noise_channel,
                         noise_dnn, denoising_conv, denoising_channel, kernel_size, use_two_dim, extra='',
                         show_name=None, dft_chuck=0):
    model = CBDNetSFModel(csi_dataloader, pilot_count, noise_level_conv=noise_level_conv,
                          noise_channel=noise_channel,
                          noise_dnn=noise_dnn, denoising_conv=denoising_conv, denoising_channel=denoising_channel,
                          kernel_size=kernel_size, use_two_dim=use_two_dim, use_true_sigma=False,
                          only_return_noise_level=True, extra=extra, dft_chuck=dft_chuck)
    model_method = InterpolationMethodModel(model, config.USE_GPU)
    xp = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ xp
    sigma_list, sigma_dict_list = [], []

    for snr in snr_list:
        sigma_dict = {}
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        sigma = (var ** 0.5).mean().item()
        sigma_list.append((sigma, '{}db'.format(snr)))
        sigma_hat_list = model_method.get_sigma_hat(y, h, xp, var, csi_dataloader.rhh)
        sigma_dict[show_name + '-{}db'.format(snr)] = sigma_hat_list
        sigma_dict_list.append(sigma_dict)

        sigma_hat_v = np.array(sigma_hat_list)
        sigma_v = np.full(sigma_hat_v.shape, sigma)
        count = sigma_v.shape[0]
        error = sigma_hat_v - sigma_v
        up_est_count = np.sum(error > 0)
        down_est_count = np.sum(error < 0)
        error_mean = np.abs(error).mean()
        error_median = np.median(error)
        error_var = np.var(error)
        max_value = np.max(np.abs(error))
        text_result = '{}db:est sigma error: mean {:.4}, up {}/{}, down {}/{}, median {:.4}, var {:.4}, max {:.4}'.format(
            snr, error_mean, up_est_count,
            count, down_est_count,
            count, error_median,
            error_var, max_value)
        logging.error(text_result)

    draw_point_and_line([i for i in range(0, csi_dataloader.n_t * h.shape[0])], sigma_dict_list, sigma_list,
                        text_label='', title='sigma-est',
                        save_dir=config.INTERPOLATION_RESULT_IMG)


def draw_relu():
    import matplotlib.pyplot as plt
    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s

    def tanh(x):
        s1 = np.exp(x) - np.exp(-x)
        s2 = np.exp(x) + np.exp(-x)
        s = s1 / s2
        return s

    def relu(x):
        s = np.where(x < 0, 0, x)
        return s

    def get_list(f, x_start, x_end, x_step=0.01):
        x = np.arange(x_start, x_end, x_step)
        ret = []
        for x_ in x:
            ret.append(f(x_))
        return x, np.array(ret)

    x = np.arange(-5, 5, 0.01)
    draw_dict = {'ReLu': [], 'Sigmoid': [], 'tanh': []}
    for x_ in x:
        draw_dict['ReLu'].append(relu(x_))
        draw_dict['Sigmoid'].append(sigmoid(x_))
        draw_dict['tanh'].append(tanh(x_))
    # draw_line(x, draw_dict, '', xlabel='x', ylabel='y',save_dir=config.INTERPOLATION_RESULT_IMG)
    plt.subplot(1, 3, 1)
    plt.plot(*get_list(relu, -5, 5))
    plt.title('ReLu')

    plt.subplot(1, 3, 2)
    plt.plot(*get_list(sigmoid, -10, 10))
    plt.title('Sigmoid')

    plt.subplot(1, 3, 3)
    plt.plot(*get_list(sigmoid, -5, 5))
    plt.title('tanh')
    plt.show()


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    # draw_relu()

    csi_dataloader = CsiDataloader('data/imt_2020_64_32_512_100.mat', train_data_radio=0.9)
    # csi_dataloader = CsiDataloader('data/spatial_mu_ULA_64_32_64_100_l10_11.mat', train_data_radio=0.9)
    # analysis_h_visualization(csi_dataloader=csi_dataloader, snr=3, )
    # analysis_window(csi_dataloader)

    # block
    cmp_model_block(csi_dataloader=csi_dataloader, snr_start=0, snr_end=17, snr_step=2, )

    # comb
    # cmp_model_comb(csi_dataloader=csi_dataloader, n_f=1, snr_start=0, snr_end=17, snr_step=2, )
    # cmp_model_comb(csi_dataloader=csi_dataloader, n_f=2, snr_start=0, snr_end=17, snr_step=2, )

    # analysis_noise_level(csi_dataloader=csi_dataloader, pilot_count=63, snr_list=[15, 20, 25], noise_level_conv=3,
    #                      noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64,
    #                      kernel_size=(3, 3), use_two_dim=True, extra='l10', show_name='CBD-SF', dft_chuck=10)
    #
    # cmp_diff_path_count('data/spatial_mu_ULA_64_32_64_10', path_list=[5, 10, 11,15,20], perfect_path=10,
    #                     snr_start=0, snr_end=26, snr_step=5, )

    # cmp_diff_pilot_count(csi_dataloader, [63, 31], snr_start=0, snr_end=30, snr_step=5, model_pilot_count=63,
    #                      noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
    #                      denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
    #                      only_return_noise_level=False, extra='l20', show_name='CBD-SF')

    # cmp_model_use_2dim(csi_dataloader=csi_dataloader, pilot_count=63, snr_start=0, snr_end=31, snr_step=5,
    #                    model_pilot_count=64, noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50),
    #                    denoising_conv=6, denoising_channel=64, kernel_size=(3, 3), use_true_sigma=True,
    #                    only_return_noise_level=False, extra='l20', show_name='CBD-SF')
