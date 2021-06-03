import torch
import logging


def normalization_3gpp_channel(H: torch.Tensor):
    """
    :param H: shape=>J,N_sc,N_r,N_t,2
    :return:
    """
    n_sc = H.shape[1]
    n_r = H.shape[2]
    n_t = H.shape[3]
    sum_h = (H ** 2).sum(-1).sum(-1).sum(-1).sum(-1)
    sum_h = sum_h.reshape(-1, 1, 1, 1, 1)
    factor = (n_r * n_t * n_sc) / sum_h
    # logging.warning('factor:{}'.format(factor.item()))
    H_normalization = H * (factor ** 0.5)
    return H_normalization


if __name__ == '__main__':
    import numpy as np
    import h5py
    from h5py import Dataset
    import os

    path_dir = '../data'
    name_list = []
    remove_list = {}
    for name in os.listdir('../data'):
        if name.endswith('.mat'):
            if '3gpp' in name:
                name_list.append(name)
            if 'normal' in name:
                remove_list[name] = True
    for name in name_list:
        if 'normal_'+name not in remove_list and 'normal' not in name:
            logging.warning('loading {}'.format(name))
            files = h5py.File(os.path.join(path_dir, name), 'r')
            H = files.get('H')
            if type(H) is not Dataset:
                H = H.get("value")
            H = torch.from_numpy(np.array(H).transpose())
            files.close()
            H_normalization = normalization_3gpp_channel(H)

            save_name = 'normal_'+name
            f = h5py.File(os.path.join(path_dir, save_name), 'w')
            f.create_dataset('H', data=H.detach().cpu().numpy().transpose())
            f.close()
            logging.warning('save normal mat:{}'.format(save_name))
        else:
            logging.warning('skip:{}'.format(name))
