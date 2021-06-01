import numpy as np


def gaussian_channel_generate(J, n_r, n_t, n_sc=1):
    h_r = np.random.normal(0, (1 / 2 / n_r) ** 0.5, [J, n_sc, n_r, n_t, 1])
    h_i = np.random.normal(0, (1 / 2 / n_r) ** 0.5, [J, n_sc, n_r, n_t, 1])
    H = np.concatenate((h_r, h_i), axis=-1)
    return H


if __name__ == '__main__':
    import h5py
    import sys
    import os

    n_r = 16
    n_t = 16
    J = 1
    n_sc = 1

    H = gaussian_channel_generate(J, n_r, n_t, n_sc)

    filename = 'gaussian_{}_{}_{}_{}.mat'.format(n_r, n_t, n_sc, J)
    save_path = os.path.join(os.path.split(sys.argv[0])[0], '../data/{}'.format(filename))
    f = h5py.File(save_path, 'w')
    f.create_dataset('H', data=H.transpose())
    f.close()
