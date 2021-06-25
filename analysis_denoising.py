import os
from typing import List

import torch

from loader import CsiDataloader, DataType
from model import DenoisingNetBaseModel, CBDNetBaseModel
from train import Train
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
                                   snr: int):
    sigma_dict = {str(m): [] for m in denoising_model_list}
    for m in denoising_model_list:
        m.set_only_return_noise_level(True)
    x = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ x
    n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
    y = hx + n
    h_ls = y @ torch.inverse(x)
    h_ls = complex2real(h_ls.reshape(-1, *h_ls.shape[-2:]))
    sigma = ((var / 2) ** 0.5).mean().item()
    for i in range(0, h_ls.shape[0], ANALYSIS_BATCH_SIZE):
        for model in denoising_model_list:
            h_ls_batch = h_ls[i:i + ANALYSIS_BATCH_SIZE]
            if use_gpu:
                h_ls_batch = h_ls_batch.cuda()
            _, sigma_hat = model(h_ls_batch, None)
            sigma_dict[str(model)].extend([s.item() for s in sigma_hat.flatten()])
    for m in denoising_model_list:
        m.set_only_return_noise_level(False)
    return sigma, sigma_dict, [i for i in range(h_ls.shape[0])]


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    # csi_dataloader = CsiDataloader('data/3gpp_16_16_64_100_10.mat', train_data_radio=0.9, factor=1)
    csi_dataloader = CsiDataloader('data/spatial_32_16_64_100.mat', train_data_radio=0.9, factor=1)
    # csi_dataloader = CsiDataloader('data/gaussian_16_16_1_100.mat', train_data_radio=0.9, factor=1)
    model = CBDNetBaseModel(csi_dataloader, 6, 6, 72)
    save_model_path = Train.get_save_path_from_model(model)
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
        model = model.double()
    else:
        logging.warning('unable load {}'.format(save_model_path))

    denoising_model = [model]
    sigma, sigma_dict, x = analysis_denoising_noise_level(csi_dataloader, denoising_model, 20)
    draw_point_and_line(x, sigma_dict, sigma, title='sigma est', save_dir=config.DENOISING_RESULT_IMG)

    denoising_methods = [DenoisingMethodMMSE(), DenoisingMethodLS(), DenoisingMethodIdealMMSE(),
                         DenoisingMethodModel(model, use_gpu)]
    # denoising_methods = [DenoisingMethodMMSE(), DenoisingMethodLS(), DenoisingMethodIdealMMSE()]

    nmse_dict, x = analysis_denoising(csi_dataloader, denoising_methods, 0, 30, 1)
    # draw_line(x, nmse_dict, lambda n: n <= 10)
    draw_line(x, nmse_dict, title='denoising-{}'.format(csi_dataloader.__str__()), save_dir=config.DENOISING_RESULT_IMG)
