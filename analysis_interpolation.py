from typing import List

import utils.config as config
from loader import CsiDataloader, DataType
from model import CBDNetSFModel
from train import load_model_from_file
from utils import DenoisingMethodMMSE, DenoisingMethodIdealMMSE, DenoisingMethodLS
from utils import InterpolationMethod, InterpolationMethodLine, InterpolationMethodModel
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
    n_sc, pilot_count = interpolation_method_list[0].n_sc, interpolation_method_list[0].pilot_count
    pilot_nmse_dict, data_nmse_dict, x = analysis_interpolation_pilot_data(csi_dataloader, interpolation_method_list,
                                                                           snr_start,
                                                                           snr_end, snr_step)
    draw_line(x, pilot_nmse_dict,
              title='interpolation-pilot{}|{}-{}'.format(pilot_count, n_sc, csi_dataloader.__str__()),
              save_dir=config.INTERPOLATION_RESULT_IMG)
    draw_line(x, data_nmse_dict,
              title='interpolation-data{}|{}-{}'.format(n_sc - pilot_count, n_sc, csi_dataloader.__str__()),
              save_dir=config.INTERPOLATION_RESULT_IMG)


def cmp_model_and_base_method(csi_dataloader: CsiDataloader, pilot_count, snr_start, snr_end, snr_step,
                              model_pilot_count, noise_level_conv, noise_channel, noise_dnn, denoising_conv,
                              denoising_channel, kernel_size, use_two_dim, use_true_sigma, only_return_noise_level,
                              extra='', show_name=None):
    model = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                          noise_channel=noise_channel,
                          noise_dnn=noise_dnn, denoising_conv=denoising_conv, denoising_channel=denoising_channel,
                          kernel_size=kernel_size, use_two_dim=use_two_dim, use_true_sigma=use_true_sigma,
                          only_return_noise_level=only_return_noise_level, extra=extra)
    model = load_model_from_file(model, use_gpu)
    if show_name:
        model.name = show_name
    interpolation_methods = [
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS(), True),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE(), True),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE(), True),
        InterpolationMethodModel(model, use_gpu, pilot_count)
    ]
    draw_pilot_and_data_nmse(csi_dataloader, interpolation_methods, snr_start=snr_start, snr_end=snr_end,
                             snr_step=snr_step)


def cmp_diff_pilot_count(csi_dataloader: CsiDataloader, pilot_count_list, snr_start, snr_end, snr_step,
                         model_pilot_count,
                         noise_level_conv, noise_channel, noise_dnn, denoising_conv, denoising_channel, kernel_size,
                         use_two_dim, use_true_sigma, only_return_noise_level, extra='', show_name=None):
    model = CBDNetSFModel(csi_dataloader, model_pilot_count, noise_level_conv=noise_level_conv,
                          noise_channel=noise_channel,
                          noise_dnn=noise_dnn, denoising_conv=denoising_conv, denoising_channel=denoising_channel,
                          kernel_size=kernel_size, use_two_dim=use_two_dim, use_true_sigma=use_true_sigma,
                          only_return_noise_level=only_return_noise_level, extra=extra)

    model = load_model_from_file(model, use_gpu)
    if show_name:
        model.name = show_name
    interpolation_methods = []
    for pilot_count in pilot_count_list:
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
                        show_name=None):
    model = None
    snr_x = None
    total_nmse_dict = {}
    for path in path_list:
        data_path = data_path_prefix + '_l{}_{}.mat'.format(path, path + 1)
        csi_dataloader = CsiDataloader(data_path, train_data_radio=0)
        if model is None:
            model = CBDNetSFModel(csi_dataloader, pilot_count, noise_level_conv=noise_level_conv,
                                  noise_channel=noise_channel,
                                  noise_dnn=noise_dnn, denoising_conv=denoising_conv,
                                  denoising_channel=denoising_channel,
                                  kernel_size=kernel_size, use_two_dim=use_two_dim, use_true_sigma=use_true_sigma,
                                  only_return_noise_level=only_return_noise_level, extra=extra)
            model = load_model_from_file(model, use_gpu)
            if show_name:
                model.name = show_name
        interpolation_methods = []
        for denosing_method in denosing_method_list:
            interpolation_method = InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, denosing_method, )
            interpolation_method.extra = '-{}p'.format(path)
            interpolation_methods.append(interpolation_method)
        interpolation_method = InterpolationMethodModel(model, use_gpu, pilot_count)
        interpolation_method.extra = '-{}p'.format(path)
        if path == perfect_path:
            interpolation_method.extra += '-same train'
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
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE()),
        InterpolationMethodModel(model_with_2dim, use_gpu, pilot_count),
        InterpolationMethodModel(model_without_2dim, use_gpu, pilot_count)
    ]
    draw_pilot_and_data_nmse(csi_dataloader, interpolation_methods, snr_start=snr_start, snr_end=snr_end,
                             snr_step=snr_step)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    cmp_diff_path_count('data/spatial_mu_ULA_32_16_64_10', path_list=[15, 20, 25], perfect_path=20, pilot_count=63,
                        snr_start=5, snr_end=30, snr_step=5, noise_level_conv=4, noise_channel=32,
                        noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64, kernel_size=(3, 3),
                        use_two_dim=True, use_true_sigma=True, only_return_noise_level=False, extra='l20',
                        show_name='CBD-SF')

    csi_dataloader = CsiDataloader('data/spatial_mu_ULA_32_16_64_10_l20_21.mat', train_data_radio=0.9)
    cmp_diff_pilot_count(csi_dataloader, [63, 31, 21], snr_start=5, snr_end=30, snr_step=5, model_pilot_count=31,
                         noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
                         denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                         only_return_noise_level=False, extra='l20', show_name='CBD-SF')

    cmp_model_and_base_method(csi_dataloader=csi_dataloader, pilot_count=64, snr_start=5, snr_end=30, snr_step=1,
                              model_pilot_count=64, noise_level_conv=4, noise_channel=32,
                              noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64, kernel_size=(3, 3),
                              use_two_dim=True, use_true_sigma=True, only_return_noise_level=False, extra='l20',
                              show_name='CBD-SF')

    cmp_model_use_2dim(csi_dataloader=csi_dataloader, pilot_count=63, snr_start=5, snr_end=30, snr_step=2,
                       model_pilot_count=64, noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50),
                       denoising_conv=6, denoising_channel=64, kernel_size=(3, 3), use_true_sigma=True,
                       only_return_noise_level=False, extra='l20', show_name='CBD-SF')
