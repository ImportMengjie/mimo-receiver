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
from utils import draw_line


def analysis_detection(csi_dataloader: CsiDataloader, detection_method_list: List[DetectionMethod], snr_start, snr_end,
                       snr_step=1, modulation='qpsk'):
    nmse_list = [[] for _ in range(len(detection_method_list))]
    x = torch.from_numpy(csi_dataloader.get_x(dataType=DataType.test, modulation=modulation))
    h = torch.from_numpy(csi_dataloader.get_h(DataType.test))
    if torch.cuda.is_available():
        x = x.cuda()
        h = h.cuda()
    hx = h @ x
    for snr in range(snr_start, snr_end, snr_step):
        n, var = csi_dataloader.noise_snr_range(hx.detach().numpy(), [snr, snr + 1], one_col=True)
        n = torch.from_numpy(n)
        var = torch.from_numpy(var)
        if torch.cuda.is_available():
            n = n.cuda()
            var = var.cuda()

        y = hx + n
        for i in range(0, len(detection_method_list)):
            nmse = detection_method_list[i].get_nmse(y, h, x, var)
            nmse_list[i].append(nmse)
    nmse_k_v = {}
    for i in range(len(nmse_list)):
        nmse_k_v[detection_method_list[i].get_key_name()] = nmse_list[i]
    return nmse_k_v, list(range(snr_start, snr_end, snr_step))


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    csi_dataloader = CsiDataloader('data/gaussian_16_16_1_1000.mat', 0.1)
    model = DetectionNetModel(csi_dataloader.n_r, csi_dataloader.n_t, 32, True, modulation='qpsk')
    save_model_path = os.path.join(Train.save_dir, model.__str__() + ".pth.tar")
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
    else:
        logging.warning('unable load {} file'.format(save_model_path))
    # detection_methods = [DetectionMethodZF('qpsk'), DetectionMethodMMSE('qpsk'), DetectionMethodModel(model, 'qpsk')]
    # detection_methods = [DetectionMethodMMSE('qpsk')]
    detection_methods = [DetectionMethodMMSE('qpsk'), DetectionMethodModel(model, 'qpsk')]
    # detection_methods = [DetectionMethodModel(model, 'qpsk')]

    nmse_dict, x = analysis_detection(csi_dataloader, detection_methods, 0, 60)
    draw_line(x, nmse_dict)
