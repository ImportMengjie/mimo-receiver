import os
from typing import List
import torch

from loader import CsiDataloader, DataType
from model import DetectionNetModel
from train import Train
from utils import DetectionMethod
from utils import DetectionZeroForce
from utils import DetectionMMSE
from utils import DetectionModel
from utils import draw_line


def analysis_detection(csi_dataloader: CsiDataloader, detection_method_list: List[DetectionMethod], snr_start, snr_end,
                       snr_step=1, modulation='qpsk'):
    nmse_list = [[] for _ in range(len(detection_method_list))]
    x = torch.from_numpy(csi_dataloader.get_x(dataType=DataType.test, modulation=modulation))
    h = torch.from_numpy(csi_dataloader.get_h(DataType.test))
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(h.shape[0], [snr, snr + 1], one_col=True)
        n = torch.from_numpy(n)
        var = torch.from_numpy(var)
        y = h@x+n

        for i in range(0, len(detection_method_list)):
            nmse = detection_method_list[i].get_nmse(y, h, x, var)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[detection_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


if __name__ == '__main__':
    csi_dataloader = CsiDataloader('data/h_16_16_64_1.mat')
    model = DetectionNetModel(csi_dataloader.n_r, csi_dataloader.n_t, 10, True, modulation='qpsk')
    save_model_path = os.path.join(Train.save_dir, model.__str__()+".pth.tar")
    model_info = torch.load(save_model_path)
    model.load_state_dict(model_info['state_dict'])
    # detection_methods = [DetectionZeroForce('qpsk'), DetectionMMSE('qpsk'), DetectionModel(model, 'qpsk')]
    # detection_methods = [DetectionMMSE('qpsk')]
    detection_methods = [DetectionMMSE('qpsk'), DetectionModel(model, 'qpsk')]

    nmse_dict, x = analysis_detection(csi_dataloader, detection_methods, 20, 200)
    draw_line(x, nmse_dict)
