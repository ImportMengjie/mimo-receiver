import os
from typing import List

import torch

from loader import CsiDataloader, DataType
from model import InterpolationNetModel, CBDNetSFModel
from train import Train
from utils import InterpolationMethod, InterpolationMethodLine, InterpolationMethodModel
from utils import draw_line
import utils.config as config
from utils import DenoisingMethodMMSE, DenoisingMethodIdealMMSE, DenoisingMethodLS, DenoisingMethod

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu


def analysis_interpolation(csi_dataloader: CsiDataloader, interpolation_method_list: List[InterpolationMethod],
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
        pilot_nmse_k_v[interpolation_method_list[i].get_key_name()] = pilot_nmse_list[i]
        data_nmse_k_v[interpolation_method_list[i].get_key_name()] = data_nmse_list[i]
    return pilot_nmse_k_v, data_nmse_k_v, list(range(snr_start, snr_end, snr_step))


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    csi_dataloader = CsiDataloader('data/spatial_mu_ULA_32_16_64_10_l20_21.mat', train_data_radio=0.9)
    pilot_count = 31
    model = CBDNetSFModel(csi_dataloader, pilot_count, noise_level_conv=4, noise_channel=32,
                          noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64,
                          kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                          only_return_noise_level=False, extra='l20')
    model.name = 'CBD-SF'
    save_model_path = Train.get_save_path_from_model(model)
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
        model = model.double().eval()
        if use_gpu:
            model = model.cuda()
    else:
        logging.warning('unable load {} model'.format(save_model_path))
    interpolation_methods = [  # InterpolationMethodLine(csi_dataloader.n_sc, pilot_count),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodLS()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodMMSE()),
        InterpolationMethodLine(csi_dataloader.n_sc, pilot_count, DenoisingMethodIdealMMSE()),
        InterpolationMethodModel(model, use_gpu)
    ]

    pilot_nmse_dict, data_nmse_dict, x = analysis_interpolation(csi_dataloader, interpolation_methods, 0, 30, 2)
    draw_line(x, pilot_nmse_dict, title='interpolation-pilot-{}'.format(csi_dataloader.__str__()),
              save_dir=config.INTERPOLATION_RESULT_IMG)
    draw_line(x, data_nmse_dict, title='interpolation-data-{}'.format(csi_dataloader.__str__()),
              save_dir=config.INTERPOLATION_RESULT_IMG)
