import os
import json

import torch

from utils import config
from utils import draw_line
from utils import DenoisingMethodLS
from loader import CsiDataloader, DataType

import numpy as np


def analysis_loss_nmse():
    for name in os.listdir(config.RESULT):
        if name.endswith('.json') and name.startswith('loss_nmse'):
            with open(os.path.join(config.RESULT, name)) as f:
                loss_data = json.load(f)
            shortname = loss_data['shortname']
            basename = loss_data.get('basename')
            loss_list = loss_data['loss']
            x = [i for i in range(0, len(loss_list))]
            draw_line(x, {shortname: loss_list}, title='loss', xlabel='iteration', ylabel='nmse(db)',
                      save_dir=config.RESULT_IMG, diff_line_markers=True)


def analysis_dft_denosing(data_path, fix_snr, max_count, path_start, path_end=None):
    csi_loader = CsiDataloader(data_path, train_data_radio=1)
    if path_end is None:
        path_end = csi_loader.n_sc + 1
    xp = csi_loader.get_pilot_x()
    h = csi_loader.get_h(DataType.train)
    hx = h @ xp
    n, var = csi_loader.noise_snr_range(hx, [fix_snr, fix_snr + 1], one_col=False)
    y = hx + n
    h_hat = DenoisingMethodLS().get_h_hat(y, h, xp, var, csi_loader.rhh)
    g_hat = h_hat.permute(0, 3, 1, 2).numpy().reshape((-1, csi_loader.n_sc, csi_loader.n_r))
    g = h.permute(0, 3, 1, 2).numpy().reshape((-1, csi_loader.n_sc, csi_loader.n_r))

    g_hat = g_hat[0:max_count]
    g = g[0:max_count]

    g_hat_idft = np.fft.ifft(g_hat, axis=-2)
    nmse_dict = {str(i): [] for i in range(g.shape[0])}
    nmse_dict.update({str(i)+'ls': [] for i in range(g.shape[0])})
    for path_count in range(path_start, path_end):
        chuck_array = np.concatenate((np.ones(path_count), np.zeros(csi_loader.n_sc - path_count))).reshape((-1, 1))
        not_chuck_array = np.concatenate((np.zeros(csi_loader.n_sc - path_count), np.ones(path_count))).reshape((-1, 1))
        g_hat_idft_chuck = g_hat_idft * chuck_array
        g_hat_idft_not_chuck = g_hat_idft * chuck_array
        g_hat_chuck = np.fft.fft(g_hat_idft_chuck, axis=-2)
        for i in range(g_hat_chuck.shape[0]):
            i_g_hat_chuck = g_hat_chuck[i]
            i_g = g[i]
            i_g_hat = g_hat[i]
            i_nmse = 10*np.log10(((np.abs(i_g-i_g_hat_chuck)**2).sum()/(np.abs(i_g)**2).sum()).mean())
            nmse_dict[str(i)].append(i_nmse)
            ls_nmse = 10*np.log10(((np.abs(i_g-i_g_hat)**2).sum()/(np.abs(i_g)**2).sum()).mean())
            nmse_dict[str(i)+'ls'].append(ls_nmse)

    x = list(range(path_start, path_end))

    draw_line(x, nmse_dict, xlabel='chuck_path', diff_line_markers=True)


if __name__ == '__main__':
    # analysis_loss_nmse()
    analysis_dft_denosing(data_path="data/3gpp_mu_32_16_512_5_5.mat", fix_snr=35, max_count=3, path_start=1,
                          path_end=50)
