from typing import List
import numpy as np

import utils.config as config
from loader import CsiDataloader, DataType
from model import CBDNetSFModel
from train import load_model_from_file
from utils import DenoisingMethodMMSE, DenoisingMethodIdealMMSE, DenoisingMethodLS, draw_point_and_line
from utils import InterpolationMethod, InterpolationMethodLine, InterpolationMethodModel, InterpolationMethodChuck
from utils import draw_line

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu

denosing_method_list = [DenoisingMethodMMSE(), DenoisingMethodIdealMMSE()]


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
    else:
        title = 'comb-interpolation-pilot{}|{}-{}'.format(pilot_count, n_sc, csi_dataloader.__str__())
    draw_line(x, pilot_nmse_dict,
              title=title,
              save_dir=config.INTERPOLATION_RESULT_IMG)
    if not is_denosing:
        draw_line(x, data_nmse_dict,
                  title='comb-interpolation-data{}|{}-{}'.format(n_sc - pilot_count, n_sc, csi_dataloader.__str__()),
                  save_dir=config.INTERPOLATION_RESULT_IMG)


def cmp_model_and_base_method(csi_dataloader: CsiDataloader, pilot_count, snr_start, snr_end, snr_step,
                              model_pilot_count, noise_level_conv, noise_channel, noise_dnn, denoising_conv,
                              denoising_channel, kernel_size, use_two_dim, use_true_sigma, only_return_noise_level,
                              extra='', show_name=None, dft_chuck=0, use_dft_padding=False):
    model = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                          noise_channel=noise_channel, noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                          denoising_channel=denoising_channel, kernel_size=kernel_size, use_two_dim=use_two_dim,
                          use_true_sigma=use_true_sigma, only_return_noise_level=only_return_noise_level, extra=extra,
                          dft_chuck=dft_chuck, use_dft_padding=use_dft_padding)
    model = load_model_from_file(model, use_gpu)
    if show_name:
        model.name = show_name
    interpolation_methods = [
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE()),
        # InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS(), True),
        InterpolationMethodChuck(csi_dataloader.n_sc, pilot_count, 10, DenoisingMethodIdealMMSE()),
        # InterpolationMethodChuck(csi_dataloader.n_sc, pilot_count, 10, DenoisingMethodMMSE()),
        # InterpolationMethodChuck(csi_dataloader.n_sc, pilot_count, 20, DenoisingMethodLS()),
        # InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE(), True),
        # InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE(), True),
        InterpolationMethodModel(model, use_gpu, pilot_count)
    ]
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


def cmp_diff_path_count(data_path_prefix, path_list, perfect_path, pilot_count, snr_start, snr_end, snr_step,
                        noise_level_conv, noise_channel, noise_dnn, denoising_conv,
                        denoising_channel, kernel_size, use_two_dim, use_true_sigma, only_return_noise_level, extra='',
                        show_name=None, dft_chuck=0, use_dft_padding=False):
    model = None
    snr_x = None
    total_nmse_dict = {}
    for path in path_list:
        data_path = data_path_prefix + '_l{}_{}.mat'.format(path, path + 1)
        csi_dataloader = CsiDataloader(data_path, train_data_radio=0)
        if model is None:
            model = CBDNetSFModel(csi_dataloader, pilot_count, noise_level_conv=noise_level_conv,
                                  noise_channel=noise_channel, noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                                  denoising_channel=denoising_channel, kernel_size=kernel_size, use_two_dim=use_two_dim,
                                  use_true_sigma=use_true_sigma, only_return_noise_level=only_return_noise_level,
                                  extra=extra, dft_chuck=dft_chuck, use_dft_padding=use_dft_padding)
            model = load_model_from_file(model, use_gpu)
            if show_name:
                model.name = show_name
        if not use_dft_padding:
            model.dft_chuck = path
        interpolation_methods = []
        interpolation_method = InterpolationMethodModel(model, use_gpu, pilot_count)
        interpolation_method.extra = '-{}p'.format(path)
        if path == perfect_path:
            interpolation_method.extra += '-same train'
        interpolation_methods.append(interpolation_method)
        # if path == perfect_path:
        #     for denosing_method in denosing_method_list:
        #         interpolation_method = InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, denosing_method, )
        #         interpolation_method.extra = '-{}p'.format(path)
        #         interpolation_methods.append(interpolation_method)
        if path == perfect_path:
            interpolation_method = InterpolationMethodChuck(csi_dataloader.n_sc, pilot_count, 20, DenoisingMethodMMSE(),
                                                            '-chuck20')
            interpolation_methods.append(interpolation_method)
        nmse_dict, snr_x = analysis_interpolation_total(csi_dataloader, interpolation_methods, snr_start, snr_end,
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


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    csi_dataloader = CsiDataloader('data/spatial_mu_ULA_64_32_64_400_l10_11.mat', train_data_radio=0.95)

    cmp_model_and_base_method(csi_dataloader=csi_dataloader, pilot_count=63, snr_start=0, snr_end=30, snr_step=2,
                              model_pilot_count=31, noise_level_conv=4, noise_channel=32,
                              noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64, kernel_size=(3, 3),
                              use_two_dim=True, use_true_sigma=True, only_return_noise_level=False, extra='l10',
                              show_name='CBD-SF', dft_chuck=10)

    analysis_noise_level(csi_dataloader=csi_dataloader, pilot_count=63, snr_list=[15, 20, 25], noise_level_conv=3,
                         noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64,
                         kernel_size=(3, 3), use_two_dim=True, extra='l10', show_name='CBD-SF', dft_chuck=10)

    cmp_diff_path_count('data/spatial_mu_ULA_64_32_64_10', path_list=[5, 10, 15, 20], perfect_path=10,
                        pilot_count=63,
                        snr_start=0, snr_end=31, snr_step=5, noise_level_conv=4, noise_channel=32,
                        noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64, kernel_size=(3, 3),
                        use_two_dim=True, use_true_sigma=True, only_return_noise_level=False, extra='l10',
                        show_name='CBD-SF')

    # cmp_diff_pilot_count(csi_dataloader, [63, 31], snr_start=0, snr_end=30, snr_step=5, model_pilot_count=63,
    #                      noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
    #                      denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
    #                      only_return_noise_level=False, extra='l20', show_name='CBD-SF')

    # cmp_model_use_2dim(csi_dataloader=csi_dataloader, pilot_count=63, snr_start=0, snr_end=31, snr_step=5,
    #                    model_pilot_count=64, noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50),
    #                    denoising_conv=6, denoising_channel=64, kernel_size=(3, 3), use_true_sigma=True,
    #                    only_return_noise_level=False, extra='l20', show_name='CBD-SF')
