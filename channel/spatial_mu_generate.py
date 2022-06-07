import numpy as np
from spatial_generate import Arrayant, get_steering_vector


def get_freq_h(N_r_v, N_r_h, N_t, K, L_range: list, low_freq, sc_band, arrayant: Arrayant, path_count):
    assert arrayant == Arrayant.ULA
    LightSpeed = 299792458
    N_r = N_r_v * N_r_h
    path_gain_var = 1
    H = np.zeros((K, N_r, N_t), dtype=np.complex_)
    lambda_list = LightSpeed / (low_freq + np.arange(0, K) * sc_band)
    d = (LightSpeed / low_freq) / 2

    angle_min = -np.pi / 2
    angle_max = np.pi / 2

    for i in range(0, N_t):
        L = np.random.randint(*L_range)
        path_count.append(L)
        l_v = np.arange(0, L).reshape((-1, 1, 1))
        path_gain = np.random.normal(0, np.sqrt(path_gain_var / 2), (L, 1, 1)) + 1j * np.random.normal(0, np.sqrt(
            path_gain_var / 2), (L, 1, 1))
        # path_gain = np.sort(path_gain, 0)[::-1]
        arrival_angle_azi = angle_min + np.random.random((L, 1, 1)) * (angle_max - angle_min)
        for k in range(0, K):
            steering_vector = get_steering_vector(N_r_v, N_r_h, arrival_angle_azi, None, lambda_list[k], d, arrayant)
            gain = path_gain * steering_vector
            W_k_l = np.e ** (-1j * 2 * np.pi * l_v * k / K)
            H[k, :, i] = (np.sqrt(N_r / L) * (W_k_l * gain).sum(0, keepdims=True)).squeeze()
    return H


def spatial_mu_generate(N_r, N_t, K, J, L_range, low_freq, sc_band, arrayant: Arrayant):
    N_r_v, N_r_h = N_r
    H_list = []
    path_count = []
    for j in range(J):
        H = get_freq_h(N_r_v, N_r_h, N_t, K, L_range, low_freq, sc_band, arrayant, path_count)
        H_r = H.real.reshape(H.shape + (1,))
        H_i = H.imag.reshape(H.shape + (1,))
        H = np.concatenate((H_r, H_i), -1)
        H_list.append(H.reshape((1,) + H.shape))
    return np.concatenate([H for H in H_list], 0), path_count


if __name__ == '__main__':
    import os
    import sys
    import h5py

    J = 10
    K = 64
    N_r = 8, 8
    N_t = 32
    L_range = (20, 21)
    low_freq = 2.4e9
    sc_band = 2e4
    arrayant = Arrayant.ULA

    H, path_count = spatial_mu_generate(N_r, N_t, K, J, L_range, low_freq, sc_band, arrayant)

    n_r = N_r[0] * N_r[1]
    n_t = N_t
    n_sc = K
    filename = 'spatial_mu_{}_{}_{}_{}_{}_l{}_{}.mat'.format(arrayant.name, n_r, n_t, n_sc, J, L_range[0], L_range[1])
    save_path = os.path.join(os.path.split(sys.argv[0])[0], '../data/{}'.format(filename))
    f = h5py.File(save_path, 'w')
    f.create_dataset('H', data=H.transpose())
    f.create_dataset('path_count', data=np.array(path_count))
    f.close()
