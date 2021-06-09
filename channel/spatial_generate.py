import numpy as np


def get_steering_vector(n_v, n_h, azimuth, elevation, lambda_):
    d = lambda_ / 2
    n_v_v = np.arange(0, n_v).reshape((-1, 1))
    n_h_v = np.arange(0, n_h).reshape((-1, 1))
    azimuth_v = np.e ** (n_v_v * -1j * 2 * np.pi * d * np.sin(azimuth) * np.sin(elevation) / lambda_)
    elevation_v = np.e ** (n_h_v * -1j * 2 * np.pi * d * np.cos(elevation) / lambda_)
    # steering_v = np.kron(azimuth_v, elevation_v) / np.sqrt(n_v * n_h)
    steering_v = np.concatenate(
        [np.kron(azimuth_v[i], elevation_v[i]).reshape((1, -1, 1)) for i in range(azimuth_v.shape[0])], 0)
    steering_v = steering_v / np.sqrt(n_v * n_h)
    return steering_v


def get_freq_h(N_r_v, N_r_h, N_t_v, N_t_h, K, L, low_freq, sc_band):
    LightSpeed = 299792458
    N_r = N_r_v * N_r_h
    N_t = N_t_v * N_t_h
    path_gain_var = 1
    H = np.zeros((K, N_r, N_t), dtype=np.complex_)
    lambda_list = (low_freq + np.arange(0, K) * sc_band) / LightSpeed

    angle_min = -np.pi / 3
    angle_max = np.pi / 3

    arrival_angle_azi = angle_min + np.random.random((L, 1, 1)) * (angle_max - angle_min)
    arrival_angle_ele = angle_min + np.random.random((L, 1, 1)) * (angle_max - angle_min)
    departure_angle_azi = angle_min + np.random.random((L, 1, 1)) * (angle_max - angle_min)
    departure_angle_ele = angle_min + np.random.random((L, 1, 1)) * (angle_max - angle_min)
    path_gain = np.random.normal(0, np.sqrt(path_gain_var / 2), (L, 1, 1)) + 1j * np.random.normal(0, np.sqrt(
        path_gain_var / 2), (L, 1, 1))
    path_gain = np.sort(path_gain, 0)[::-1]
    l_v = np.arange(0, L).reshape((-1, 1, 1))
    for k in range(0, K):
        arrival = get_steering_vector(N_r_v, N_r_h, arrival_angle_azi, arrival_angle_ele, lambda_list[k])
        departure = get_steering_vector(N_t_v, N_t_h, departure_angle_azi, departure_angle_ele, lambda_list[k])
        gain = path_gain * arrival @ departure.swapaxes(-1, -2)
        W_k_l = np.e ** (-1j * 2 * np.pi * l_v * k / K)
        H[k] = np.sqrt(N_r * N_t / L) * (W_k_l * gain).sum(0, keepdims=True)
    return H


def spatial_generate(N_r, N_t, K, J, L_range, low_freq, sc_band):
    N_r_v, N_r_h = N_r
    N_t_v, N_t_h = N_t
    H_list = []
    L = np.random.randint(*L_range, size=(J,))
    for i in range(J):
        H = get_freq_h(N_r_v, N_r_h, N_t_v, N_t_h, K, L=L[i], low_freq=low_freq, sc_band=sc_band)
        H_r = H.real.reshape(H.shape + (1,))
        H_i = H.imag.reshape(H.shape + (1,))
        H = np.concatenate((H_r, H_i), -1)
        H_list.append(H.reshape((1,) + H.shape))
    return np.concatenate([H for H in H_list], 0)


if __name__ == '__main__':
    import os
    import sys
    import h5py

    J = 100
    K = 64
    N_r = 4, 4
    N_t = 4, 4
    L_range = (4, 5)
    low_freq = 2.4e9
    sc_band = 2e4
    H = spatial_generate(N_r, N_t, K, J, L_range, low_freq, sc_band)

    n_r = N_r[0] * N_r[1]
    n_t = N_t[0] * N_t[1]
    n_sc = K
    filename = 'spatial_{}_{}_{}_{}.mat'.format(n_r, n_t, n_sc, J)
    save_path = os.path.join(os.path.split(sys.argv[0])[0], '../data/{}'.format(filename))
    f = h5py.File(save_path, 'w')
    f.create_dataset('H', data=H.transpose())
    f.close()
