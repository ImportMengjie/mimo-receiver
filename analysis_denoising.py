import os
from typing import List

import torch

from loader import CsiDataloader, DataType
from model import DenoisingNetModel
from train import Train
from utils import DenoisingMethod, draw_line
from utils import DenoisingMethodLS
from utils import DenoisingMethodMMSE
from utils import DenoisingMethodModel

import utils.config as config

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu


def analysis_denoising(csi_dataloader: CsiDataloader, denoising_method_list: List[DenoisingMethod], snr_start, snr_end,
                       snr_step=1):
    nmse_list = [[] for _ in range(len(denoising_method_list))]
    x = csi_dataloader.get_pilot_x()
    h = csi_dataloader.get_h(DataType.test)
    hx = h @ x
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=False)
        y = hx + n
        for i in range(len(denoising_method_list)):
            nmse = denoising_method_list[i].get_nmse(y, h, x, var)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[denoising_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    csi_dataloader = CsiDataloader('data/3gpp_16_16_64_100_10.mat', train_data_radio=0.9, factor=1)
    model = DenoisingNetModel(csi_dataloader)
    save_model_path = Train.get_save_path_from_model(model)
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
    else:
        logging.warning('unable load {}'.format(save_model_path))
    detection_methods = [DenoisingMethodLS(), DenoisingMethodMMSE(), DenoisingMethodModel(model, use_gpu)]
    # detection_methods = [DenoisingMethodMMSE(), DenoisingMethodLS()]

    nmse_dict, x = analysis_denoising(csi_dataloader, detection_methods, 2, 80, 10)
    # draw_line(x, nmse_dict, lambda n: n <= 10)
    draw_line(x, nmse_dict, title='denoising-{}'.format(csi_dataloader.__str__()), save_dir=config.DENOISING_RESULT_IMG)
