import os.path

import numpy as np
import h5py
import torch
from h5py import Dataset
import scipy.io as scio


def un_squeeze(H):
    if len(H.shape) < 5:
        return H.reshape(H.shape + (1,))
    return H


def get_freq_G(N_sc, H: np.ndarray, path_count):
    N_r = H.shape[0]
    path = H.shape[2]
    j = H.shape[-1]
    path_count.extend([path] * H.shape[3])
    G_m_s = np.zeros((N_sc, N_r, j), dtype=np.complex_)
    l_v = np.arange(0, path).reshape((-1, 1))
    H = H.squeeze()
    for k in range(0, N_sc):
        W_k_l = np.e ** (-1j * 2 * np.pi * l_v * k / N_sc)
        G_m_s[k] = (H * W_k_l).sum(-2, keepdims=True).squeeze()
    return G_m_s


def imt_2020_generate(N_sc: int, step=2):
    H = None
    sim = 1
    path_count = []
    J, N_r, N_t = None, None, None
    while os.path.exists('H/H_sim{}_data.mat'.format(sim)):
        data = scio.loadmat('H/H_sim{}_data.mat'.format(sim))
        h_los = un_squeeze(data.get('H_u_s_n_LOS'))
        h_nlos = un_squeeze(data.get('H_u_s_n_NLOS'))
        h_o2i = un_squeeze(data.get('H_u_s_n_O2I'))

        J = h_los.shape[3]
        N_r = h_los.shape[0]
        N_t = h_los.shape[4] + h_nlos.shape[4] + h_o2i.shape[4]
        if H is None:
            H = np.zeros((0, N_sc, N_r, N_t), dtype=np.complex_)
        for i in range(0, J, step):
            g_los = get_freq_G(N_sc, h_los[:, :, :, i], path_count)
            g_nlos = get_freq_G(N_sc, h_nlos[:, :, :, i], path_count)
            g_o2i = get_freq_G(N_sc, h_o2i[:, :, :, i], path_count)
            H_i = np.concatenate((g_los, g_nlos, g_o2i), -1)
            H_i = H_i.reshape((1,) + H_i.shape)
            H = np.concatenate((H, H_i), axis=0)
        sim += 1
    H_r = H.real.reshape(H.shape + (1,))
    H_i = H.imag.reshape(H.shape + (1,))
    H = np.concatenate((H_r, H_i), -1)
    return H, path_count, H.shape[0], N_r, N_t,


if __name__ == '__main__':
    import sys

    N_sc = 64
    H, path_count, J, N_r, N_t = imt_2020_generate(N_sc, 2)
    if H is not None:
        filename = 'imt_2020_{}_{}_{}_{}.mat'.format(N_r, N_t, N_sc, J)
        path_count = np.array(path_count)
        save_path = os.path.join(os.path.split(sys.argv[0])[0], '../../data/{}'.format(filename))
        f = h5py.File(save_path, 'w')
        f.create_dataset('H', data=H.transpose())
        f.create_dataset('path_count', data=path_count)
        f.close()
