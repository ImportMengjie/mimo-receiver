import os
from typing import List
import torch

from loader import CsiDataloader, DataType
from model import DetectionNetModel
from train import Train
from utils import DetectionMethod
from utils import DetectionMethodZF
from utils import DetectionMethodMMSE
from utils import DetectionMethodModel
from utils import DetectionMethodConjugateGradient
from utils import draw_line


def analysis_detection_nmse(csi_dataloader: CsiDataloader, detection_method_list: List[DetectionMethod], snr_start,
                            snr_end,
                            snr_step=1, modulation='bpsk', dataType=DataType.test):
    nmse_list = [[] for _ in range(len(detection_method_list))]
    x = csi_dataloader.get_x(dataType=dataType, modulation=modulation)
    h = csi_dataloader.get_h(dataType)
    hx = h @ x
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=True)

        y = hx + n
        for i in range(0, len(detection_method_list)):
            nmse = detection_method_list[i].get_nmse(y, h, x, var)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[detection_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


def analysis_detection_layer(csi_dataloader: CsiDataloader, model_list: [DetectionNetModel], fix_snr=30,
                             modulation='bpsk', dataType=DataType.test):
    x = csi_dataloader.get_x(dataType=dataType, modulation=modulation)
    h = csi_dataloader.get_h(dataType)
    hx = h @ x
    n, var = csi_dataloader.noise_snr_range(hx, [fix_snr, fix_snr + 1], one_col=True)
    y = hx + n
    model_method_list = [DetectionMethodModel(model, modulation) for model in model_list]
    nmse_k_v = {}
    mmse_method = DetectionMethodMMSE(modulation)
    cj_method = DetectionMethodConjugateGradient(modulation, 1)
    iter_list = []
    for layer in range(1, csi_dataloader.n_t*2+1):
        iter_list.append(layer)
        for method in model_method_list:
            method.model.set_test_layer(layer)
            nmse = method.get_nmse(y, h, x, var)
            nmses = nmse_k_v.get(method.get_key_name(), [])
            nmses.append(nmse)
            nmse_k_v[method.get_key_name()] = nmses
        cj_method.iterate = layer
        nmses = nmse_k_v.get(cj_method.get_key_name_short(), [])
        nmses.append(cj_method.get_nmse(y, h, x, var))
        nmse_k_v[cj_method.get_key_name_short()] = nmses

        mmse_method.get_nmse(y, h, x, var)
        nmses = nmse_k_v.get(mmse_method.get_key_name(), [])
        nmses.append(mmse_method.get_nmse(y, h, x, var))
        nmse_k_v[mmse_method.get_key_name()] = nmses
    return nmse_k_v, iter_list


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    csi_dataloader = CsiDataloader('data/gaussian_16_16_1_1.mat', train_data_radio=0, factor=10000)
    layer = csi_dataloader.n_t * 2
    constellation = 'bpsk'
    model = DetectionNetModel(csi_dataloader, layer, True, modulation=constellation)
    save_model_path = os.path.join(Train.save_dir, model.__str__() + ".pth.tar")
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
    else:
        logging.warning('unable load {} file'.format(save_model_path))
    detection_methods = [DetectionMethodZF(constellation), DetectionMethodMMSE(constellation),
                         DetectionMethodModel(model, constellation)]
    # detection_methods = [DetectionMethodMMSE('qpsk')]
    # detection_methods = [DetectionMethodMMSE(constellation), DetectionMethodModel(model, constellation),
    #                      DetectionMethodConjugateGradient(constellation, csi_dataloader.n_t),
    #                      DetectionMethodConjugateGradient(constellation, csi_dataloader.n_t * 2)]
    # detection_methods = [DetectionMethodModel(model, constellation)]

    nmse_dict, x = analysis_detection_nmse(csi_dataloader, detection_methods, 5, 60, modulation=constellation)
    draw_line(x, nmse_dict, title='Detection-{}'.format(csi_dataloader.__str__()))

    nmse_dict, iter_list = analysis_detection_layer(csi_dataloader, [model], 30, 'bpsk')
    draw_line(iter_list, nmse_dict, title='Detection-{}-iter'.format(csi_dataloader), xlabel='iter')
