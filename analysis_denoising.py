import os
from typing import List

import numpy as np
import torch

from loader import CsiDataloader, DataType
from model import DenoisingNetBaseModel, CBDNetBaseModel
from train import Train, load_model_from_file
from utils import DenoisingMethod, draw_line, conj_t, complex2real, draw_point_and_line
from utils import DenoisingMethodLS
from utils import DenoisingMethodMMSE
from utils import DenoisingMethodIdealMMSE
from utils import DenoisingMethodModel
from utils.config import *

import utils.config as config

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu


def analysis_denoising(csi_dataloader: CsiDataloader, denoising_method_list: List[DenoisingMethod], snr_start, snr_end,
                       snr_step=1):
    nmse_list = [[] for _ in range(len(denoising_method_list))]
    x = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    rhh = csi_dataloader.rhh
    hx = h @ x
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        for i in range(len(denoising_method_list)):
            nmse = denoising_method_list[i].get_nmse(y, h, x, var, rhh)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[denoising_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


def analysis_denoising_noise_level(csi_dataloader: CsiDataloader, denoising_model_list: List[DenoisingNetBaseModel],
                                   snr_list: List):
    for m in denoising_model_list:
        m.set_only_return_noise_level(True)
    x = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ x
    sigma_list = []
    sigma_dict_list = []
    text_label = ""
    for snr in snr_list:
        sigma_dict = {m.name: [] for m in denoising_model_list}
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        h_ls = y @ torch.inverse(x)
        h_ls = complex2real(h_ls.reshape(-1, *h_ls.shape[-2:]))
        sigma = ((var / 2) ** 0.5).mean().item()
        sigma_list.append((sigma, '{}db'.format(snr)))
        for i in range(0, h_ls.shape[0], ANALYSIS_BATCH_SIZE):
            for model in denoising_model_list:
                h_ls_batch = h_ls[i:i + ANALYSIS_BATCH_SIZE]
                if use_gpu:
                    h_ls_batch = h_ls_batch.cuda()
                _, sigma_hat = model(h_ls_batch, None)
                if sigma_hat.is_cuda:
                    sigma_hat = sigma_hat.cpu()
                sigma_dict[model.name].extend([s.item() for s in sigma_hat.flatten()])
        for m in denoising_model_list:
            m.set_only_return_noise_level(False)
        for k, v in sigma_dict.items():
            sigma_hat_v = np.array(v)
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
            draw_text = '{}db:mean{:.2}var{:.2}max{:.2}'.format(snr, error_mean, error_var, max_value)
            text_label += draw_text
        sigma_dict_list.append(sigma_dict)
    return sigma_list, sigma_dict_list, [i for i in range(h_ls.shape[0])], text_label


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    # csi_dataloader = CsiDataloader('data/3gpp_16_16_64_100_10.mat', train_data_radio=0.9, factor=1)
    csi_dataloader = CsiDataloader('data/spatial_ULA_32_16_64_100.mat', train_data_radio=0.9, factor=1)
    # csi_dataloader = CsiDataloader('data/gaussian_16_16_1_100.mat', train_data_radio=0.9, factor=1)
    model = CBDNetBaseModel(csi_dataloader, noise_level_conv_num=4, noise_channel_num=32,
                            denosing_conv_num=6, denosing_channel_num=32,
                            use_true_sigma=False, only_return_noise_level=False, extra='')
    model = load_model_from_file(model, use_gpu)

    denoising_model = [model]
    sigma_list, sigma_dict_list, x, text_label = analysis_denoising_noise_level(csi_dataloader, denoising_model,
                                                                               [15, 20])
    draw_point_and_line(x, sigma_dict_list, sigma_list, text_label=text_label, title='sigma-est',
                        save_dir=config.DENOISING_RESULT_IMG)

    denoising_methods = [DenoisingMethodMMSE(), DenoisingMethodLS(), DenoisingMethodIdealMMSE(),
                         DenoisingMethodModel(model, use_gpu)]
    # denoising_methods = [DenoisingMethodMMSE(), DenoisingMethodLS(), DenoisingMethodIdealMMSE()]

    nmse_dict, x = analysis_denoising(csi_dataloader, denoising_methods, 0, 30, 1)
    draw_line(x, nmse_dict, title='denoising-{}'.format(csi_dataloader.__str__()), save_dir=config.DENOISING_RESULT_IMG)
