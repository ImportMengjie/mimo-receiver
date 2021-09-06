import os
import json
from typing import List
import torch

from loader import CsiDataloader, DataType, DetectionNetDataset
from model import DetectionNetModel, DetectionNetLoss, DetectionNetTee
from train import Train, TrainParam, load_model_from_file
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


def analysis_detection_layer(csi_dataloader: CsiDataloader, model_list: [DetectionNetModel], fix_snr=30, max_layer=None,
                             modulation='bpsk', dataType=DataType.test):
    if max_layer is None:
        max_layer = csi_dataloader.n_t * 2 + 1
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
    for layer in range(1, max_layer):
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


def cmp_base_model_nmse_ber(csi_dataloader: CsiDataloader, snr_start, snr_end, snr_step, modulation, layer, is_vector,
                            extra='', show_name=None):
    model = DetectionNetModel(csi_dataloader, layer_nums=layer, vector=is_vector, is_training=False,
                              modulation=modulation, extra=extra)
    model = load_model_from_file(model, use_gpu)
    if show_name is not None:
        model.name = show_name
    detection_methods = [DetectionMethodZF(modulation), DetectionMethodMMSE(modulation),
                         DetectionMethodModel(model, modulation, use_gpu)]
    nmse_dict, x = analysis_detection_nmse(csi_dataloader, detection_methods, snr_start, snr_end, snr_step,
                                           modulation=modulation)
    draw_line(x, nmse_dict, title='detection-{}-{}'.format(modulation, csi_dataloader.__str__()))

    ber_dict, x = analysis_detection_ber(csi_dataloader, detection_methods, snr_start, snr_end, snr_step,
                                         modulation=modulation)
    draw_line(x, ber_dict, title='detection-{}-{}'.format(modulation, csi_dataloader.__str__()), ylabel='ber')


def cmp_diff_layers_nmse(csi_dataloader: CsiDataloader, load_data_from_files, fix_snr, max_layers, modulation, layer,
                         is_vector, extra=''):
    model = DetectionNetModel(csi_dataloader, layer_nums=layer, vector=is_vector, is_training=True,
                              modulation=modulation, extra=extra)
    save_data_name = os.path.join(config.DENOISING_RESULT, '{}.json'.format(model.__str__()))
    if not load_data_from_files or not os.path.exists(save_data_name):
        # model = load_model_from_file(model, use_gpu)
        nmse_dict, iter_list = analysis_detection_layer(csi_dataloader, [model], fix_snr, max_layers, 'bpsk')
        with open(save_data_name, 'w') as f:
            json.dump({nmse_dict: nmse_dict, 'iter_list': iter_list}, f)
    else:
        with open(save_data_name) as f:
            data = json.load(f)
            nmse_dict = data["nmse_dict"]
            iter_list = data['iter_list']
    draw_line(iter_list, nmse_dict, title='detection-{}-iter'.format(csi_dataloader), xlabel='iter/layer',
              save_dir=config.DETECTION_RESULT_IMG)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    csi_dataloader = CsiDataloader('data/spatial_mu_ULA_64_32_64_400_l10_11.mat', train_data_radio=0.9, factor=1)
    cmp_base_model_nmse_ber(csi_dataloader=csi_dataloader, snr_start=2, snr_end=25, snr_step=2, modulation='qpsk',
                            layer=32, is_vector=True, extra='', show_name='lcg-net')
    # cmp_diff_layers_nmse(csi_dataloader=csi_dataloader, load_data_from_files=True, fix_snr=15, max_layers=32,
    #                      modulation='bpsk', layer=32, is_vector=True, extra='')
