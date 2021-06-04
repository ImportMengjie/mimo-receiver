import os
from typing import List
import torch

from loader import CsiDataloader, DataType, DetectionNetDataset
from model import DetectionNetModel, DetectionNetLoss, DetectionNetTee
from train import Train, TrainParam
from utils import DetectionMethod
from utils import DetectionMethodZF
from utils import DetectionMethodMMSE
from utils import DetectionMethodModel
from utils import DetectionMethodConjugateGradient
from utils import draw_line

import utils.config as config

use_gpu = True and config.USE_GPU
config.USE_GPU = use_gpu


def analysis_detection_nmse(csi_dataloader: CsiDataloader, detection_method_list: List[DetectionMethod], snr_start,
                            snr_end,
                            snr_step=1, modulation='bpsk', dataType=DataType.test):
    nmse_list = [[] for _ in range(len(detection_method_list))]
    x, _ = csi_dataloader.get_x(dataType=dataType, modulation=modulation)
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


def analysis_detection_ber(csi_dataloader: CsiDataloader, detection_method_list: List[DetectionMethod], snr_start,
                           snr_end,
                           snr_step=1, modulation='bpsk', dataType=DataType.test):
    ber_list = [[] for _ in range(len(detection_method_list))]
    x, x_idx = csi_dataloader.get_x(dataType=dataType, modulation=modulation)
    h = csi_dataloader.get_h(dataType)
    hx = h @ x
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx, [snr, snr + 1], one_col=True)
        y = hx + n
        detection_method_list[0].get_ber(y, h, x, x_idx, var)
        for i in range(0, len(detection_method_list)):
            ber = detection_method_list[i].get_ber(y, h, x, x_idx, var)
            ber_list[i].append(ber)
    ber_k_v = {}
    for i in range(len(ber_list)):
        ber_k_v[detection_method_list[i].get_key_name()] = ber_list[i]
    return ber_k_v, list(range(snr_start, snr_end, snr_step))


def analysis_detection_layer(csi_dataloader: CsiDataloader, model_list: [DetectionNetModel], fix_snr=30,
                             modulation='bpsk', dataType=DataType.test):
    x, _ = csi_dataloader.get_x(dataType=dataType, modulation=modulation)
    h = csi_dataloader.get_h(dataType)
    hx = h @ x
    n, var = csi_dataloader.noise_snr_range(hx, [fix_snr, fix_snr + 1], one_col=True)
    y = hx + n

    param = TrainParam()
    param.epochs = 100
    criterion = DetectionNetLoss()
    dataset = DetectionNetDataset(csi_dataloader, dataType, [fix_snr, fix_snr + 1], modulation)
    train_list = [
        Train(param, dataset, model.cuda() if config.USE_GPU else model, criterion, DetectionNetTee, dataset) for
        model in model_list]

    model_method_list = [DetectionMethodModel(model, modulation, use_gpu) for model in model_list]
    nmse_k_v = {}
    mmse_method = DetectionMethodMMSE(modulation)
    cj_method = DetectionMethodConjugateGradient(modulation, 1)
    iter_list = []
    for layer in range(1, csi_dataloader.n_t * 2 + 1):
        iter_list.append(layer)
        for method, train in zip(model_method_list, train_list):
            method.model.set_training_layer(layer, False)
            train.train(save=False, reload=False, ext_log='model:{},layer:{}'.format(method.model, layer))
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
    modulation = 'bpsk'
    model = DetectionNetModel(csi_dataloader, layer, True, modulation=modulation)
    save_model_path = Train.get_save_path_from_model(model)
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
    else:
        logging.warning('unable load {} file'.format(save_model_path))
    detection_methods = [DetectionMethodZF(modulation), DetectionMethodMMSE(modulation),
                         DetectionMethodModel(model, modulation, use_gpu)]
    # detection_methods = [DetectionMethodMMSE('qpsk')]
    # detection_methods = [DetectionMethodMMSE(constellation), #DetectionMethodModel(model, constellation),
    #                      DetectionMethodConjugateGradient(constellation, csi_dataloader.n_t),
    #                      DetectionMethodConjugateGradient(constellation, csi_dataloader.n_t * 2)]
    # detection_methods = [DetectionMethodModel(model, constellation)]

    nmse_dict, x = analysis_detection_nmse(csi_dataloader, detection_methods, 0, 40, 2, modulation=modulation)
    draw_line(x, nmse_dict, title='Detection-{}'.format(csi_dataloader.__str__()))

    ber_dict, x = analysis_detection_ber(csi_dataloader, detection_methods, 0, 20, 2, modulation=modulation)
    draw_line(x, ber_dict, title='Detection-{}'.format(csi_dataloader.__str__()), ylabel='ber')
    # nmse_dict, iter_list = analysis_detection_layer(csi_dataloader, [model], 30, 'bpsk')
    # draw_line(iter_list, nmse_dict, title='Detection-{}-iter'.format(csi_dataloader), xlabel='iter/layer',
    #           save_dir=config.DETECTION_RESULT_IMG)
