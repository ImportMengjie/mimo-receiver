import os
from typing import List

import torch

from loader import CsiDataloader, DataType

from model import InterpolationNetModel
from train import Train

from utils import InterpolationMethod, InterpolationMethodLine, InterpolationMethodModel
from utils import draw_line


def analysis_interpolation(csi_dataloader: CsiDataloader, interpolation_method_list: List[InterpolationMethod],
                           snr_start, snr_end, snr_step):
    nmse_list = [[] for _ in range(len(interpolation_method_list))]
    xp = torch.from_numpy(csi_dataloader.get_pilot_x())
    h = torch.from_numpy(csi_dataloader.get_h(DataType.test))
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(h.shape[0], [snr, snr + 1], one_col=False)
        n = torch.from_numpy(n)
        var = torch.from_numpy(var)
        y = h @ xp + n
        for i in range(len(interpolation_method_list)):
            nmse = interpolation_method_list[i].get_nmse(y, h, xp, var)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[interpolation_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


if __name__ == '__main__':
    csi_dataloader = CsiDataloader('data/h_16_16_64_1.mat')
    model = InterpolationNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_sc,4)
    save_model_path = os.path.join(Train.save_dir, model.__str__() + ".pth.tar")
    model_info = torch.load(save_model_path)
    model.load_state_dict(model_info['state_dict'])
    interpolation_methods = [InterpolationMethodLine(csi_dataloader.n_sc, model.pilot_count),
                             InterpolationMethodModel(model)]

    nmse_dict, x = analysis_interpolation(csi_dataloader, interpolation_methods, 150, 200, 10)
    # draw_line(x, nmse_dict, lambda n: n <= 10)
    draw_line(x, nmse_dict, )
