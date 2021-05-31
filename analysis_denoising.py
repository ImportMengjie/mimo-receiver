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
    csi_dataloader = CsiDataloader('data/gaussian_16_16_1_1000.mat', 0)
    model = DenoisingNetModel(csi_dataloader.n_r, csi_dataloader.n_t)
    save_model_path = os.path.join(Train.save_dir, model.__str__() + ".pth.tar")
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path)
        model.load_state_dict(model_info['state_dict'])
    else:
        logging.warning('unable load {}'.format(save_model_path))
    detection_methods = [DenoisingMethodLS(), DenoisingMethodMMSE(), DenoisingMethodModel(model)]
    # detection_methods = [DenoisingMethodMMSE(), DenoisingMethodLS()]

    nmse_dict, x = analysis_denoising(csi_dataloader, detection_methods, 0, 60, 2)
    # draw_line(x, nmse_dict, lambda n: n <= 10)
    draw_line(x, nmse_dict, )
