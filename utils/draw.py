from typing import List

import matplotlib.pyplot as plt
import os

plt.rcParams['axes.unicode_minus'] = False


def draw_line(x, y_dict: dict, title=None, filter_func=None, save_dir=None, show=True, xlabel='snr(db)',
              ylabel='nmse(db)', diff_line_with=True):
    if filter_func is None:
        filter_func = lambda n: True
    lw = 1
    for name, y in y_dict.items():
        y = list(map(lambda n: n if filter_func(n) else None, y))
        plt.plot(x, y, label=name, lw=lw)
        if diff_line_with:
            lw += 1
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir:
        save_name = (title if title else '') + '-'.join(y_dict.keys()) + '.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    plt.close('all')


def draw_point_and_line(x, y_dict_point_list: List[dict], y_line_list: List, text_list=None, title=None, save_dir=None,
                        show=True, xlabel='index',
                        ylabel='sigma'):
    for y_dict_point in y_dict_point_list:
        for name, y in y_dict_point.items():
            plt.scatter(x, y, s=1, label=name)
    for y_line, name in y_line_list:
        plt.plot(x, [y_line for _ in range(len(x))], label=name)
    if title:
        plt.title(title)
    plt.xlabel(xlabel+'\n'+'\n'.join(text_list))
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir:
        save_name = (title if title else '') + '-'.join(y_dict_point_list[0].keys()) + '.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    plt.close('all')
