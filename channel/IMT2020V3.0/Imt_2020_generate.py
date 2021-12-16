import logging
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


def get_freq_G(N_sc, H: np.ndarray, fs, delay, path_count):
    N_r = H.shape[0]
    path = H.shape[2]
    j = H.shape[-1]
    G_m_s = np.zeros((N_sc, N_r, j), dtype=np.complex_)
    l_v = (delay * fs)
    path_count.extend(np.ceil(l_v[-1]))
    # l_v = np.rint(l_v)
    H = H.squeeze(1)
    for k in range(0, N_sc):
        W_k_l = np.e ** (-1j * 2 * np.pi * l_v * k / N_sc)
        G_m_s[k] = (H * W_k_l).sum(-2, keepdims=True).squeeze(1)
    return G_m_s


def imt_2020_generate(step=2):
    max_delay = 0

    def get_max_delay(*args):
        max_delay = 0
        for delay in args:
            if len(delay.flatten()):
                max_delay = max(delay.max(), max_delay)
        return max_delay

    H = None
    sim = 1
    path_count = []
    J, N_r, N_t = None, None, None
    if not os.path.exists('ScenarioParameters/base.mat'):
        raise Exception("Can't find ScenarioParameters/base.mat")
    base_conf = scio.loadmat('ScenarioParameters/base.mat')
    N_sc = base_conf.get('N_sc').item()
    bw = base_conf.get('bw').item()
    fc = base_conf.get('fc').item() * 1e9
    pathloss = scio.loadmat('LSP/Pathloss.mat')
    sf_sigmas = pathloss.get('SF_sigma').squeeze()
    pathloss = pathloss.get('Pathloss').squeeze()
    sc_band = base_conf.get('sc_band').item()

    bound_delay = N_sc/(6*bw)
    while os.path.exists('H/H_sim{}_data.mat'.format(sim)):
        logging.info('start H/H_sim{}_data.mat'.format(sim))
        h_data = scio.loadmat('H/H_sim{}_data.mat'.format(sim))
        ssp_data = scio.loadmat('SSP/Sim_{}_data.mat'.format(sim))

        los_delay = ssp_data.get('LOS')['Delay'][0, 0].swapaxes(0, 1)
        nlos_delay = ssp_data.get('NLOS')['Delay'][0, 0].swapaxes(0, 1)
        o2i_delay = ssp_data.get('O2I')['Delay'][0, 0].swapaxes(0, 1)
        max_delay = max(get_max_delay(los_delay, nlos_delay, o2i_delay), max_delay)

        h_los = un_squeeze(h_data.get('H_u_s_n_LOS'))
        h_nlos = un_squeeze(h_data.get('H_u_s_n_NLOS'))
        h_o2i = un_squeeze(h_data.get('H_u_s_n_O2I'))

        # add ssp
        h_los_index = h_data.get('LOSlinkindex').squeeze() - 1
        h_nlos_index = h_data.get('NLOSlinkindex').squeeze() - 1
        h_o2i_index = h_data.get('O2Ilinkindex').squeeze() - 1

        los_pathloss = pathloss[h_los_index]
        los_sf_sigma = sf_sigmas[h_los_index]
        nlos_pathloss = pathloss[h_nlos_index]
        nlos_sf_sigma = sf_sigmas[h_nlos_index]
        o2i_pathloss = pathloss[h_o2i_index]
        o2i_sf_sigma = sf_sigmas[h_o2i_index]

        los_pathloss = np.random.randn(*los_pathloss.shape) * los_sf_sigma + los_pathloss
        nlos_pathloss = np.random.randn(*nlos_pathloss.shape) * nlos_sf_sigma + nlos_pathloss
        o2i_pathloss = np.random.randn(*o2i_pathloss.shape) * o2i_sf_sigma + o2i_pathloss
        los_pathloss = np.sqrt(np.power(10, -los_pathloss / 10))
        nlos_pathloss = np.sqrt(np.power(10, -nlos_pathloss / 10))
        o2i_pathloss = np.sqrt(np.power(10, -o2i_pathloss / 10))
        # h_los = los_pathloss * h_los
        # h_nlos = nlos_pathloss * h_nlos
        # h_o2i = o2i_pathloss * h_o2i

        J = h_los.shape[3]
        N_r = h_los.shape[0]
        N_t = h_los.shape[4] + h_nlos.shape[4] + h_o2i.shape[4]
        if H is None:
            H = np.zeros((0, N_sc, N_r, N_t), dtype=np.complex_)
        for i in range(0, J, step):
            g_los = get_freq_G(N_sc, h_los[:, :, :, i], fs=bw, delay=los_delay, path_count=path_count)
            g_nlos = get_freq_G(N_sc, h_nlos[:, :, :, i], fs=bw, delay=nlos_delay, path_count=path_count)
            g_o2i = get_freq_G(N_sc, h_o2i[:, :, :, i], fs=bw, delay=o2i_delay, path_count=path_count)
            H_i = np.concatenate((g_los, g_nlos, g_o2i), -1)
            H_i = H_i.reshape((1,) + H_i.shape)
            H = np.concatenate((H, H_i), axis=0)
        sim += 1
    H_r = H.real.reshape(H.shape + (1,))
    H_i = H.imag.reshape(H.shape + (1,))
    H = np.concatenate((H_r, H_i), -1)
    return H, path_count, H.shape[0], N_r, N_t, N_sc, max_delay


if __name__ == '__main__':
    import sys

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    H, path_count, J, N_r, N_t, N_sc, max_delay = imt_2020_generate(2)
    if H is not None:
        filename = 'imt_2020_{}_{}_{}_{}.mat'.format(N_r, N_t, N_sc, J)
        path_count = np.array(path_count)
        save_path = os.path.join(os.path.split(sys.argv[0])[0], '../../data/{}'.format(filename))
        f = h5py.File(save_path, 'w')
        f.create_dataset('H', data=H.transpose())
        f.create_dataset('path_count', data=path_count)
        f.close()
        logging.info('save {}'.format(filename))
