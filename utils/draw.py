from typing import List

import matplotlib.pyplot as plt
import os
import json

plt.rcParams['axes.unicode_minus'] = False

line_style_list = ['-', '--', '-.', ':'][::-1]

"""
's' : 方块状
'o' : 实心圆
'^' : 正三角形
'v' : 反正三角形
'+' : 加好
'*' : 星号
'x' : x号
'p' : 五角星
'1' : 三脚架标记
'2' : 三脚架标记
"""
markers = ['^', 'v', '+', '*', 'o', 's', 'x', 'p', '1', '2']


def draw_line(x, y_dict: dict, title=None, filter_func=None, save_dir=None, show=True, xlabel='snr(db)',
              ylabel='nmse(db)', diff_line_style=True, diff_line_markers=False):
    if filter_func is None:
        filter_func = lambda n: True
    i = 0
    for name, y in y_dict.items():
        y = list(map(lambda n: n if filter_func(n) else None, y))
        marker = markers[i % len(markers)] if diff_line_markers else None
        plt.plot(x, y, label=name, lw=2, ls=line_style_list[i % len(line_style_list)], marker=marker, mfc='none')
        if diff_line_style:
            i += 1
    if title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir:
        save_name = (title if title else '') + '-'.join(y_dict.keys())
        save_name = save_name.replace('/', '|')
        save_name_img = save_name + '.png'
        save_name_json = save_name + '.json'
        data_json = {
            'title': title,
            'x': x,
            'y_dict': y_dict,
            'xlabel': xlabel,
            'ylable': ylabel,
            'style': diff_line_style,
            'markers': diff_line_markers
        }
        with open(save_name_json) as f:
            json.dump(data_json, f, indent=2)
        save_path = os.path.join(save_dir, save_name_img)
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    plt.close('all')


def draw_point_and_line(x, y_dict_point_list: List[dict], y_line_list: List, text_label=None, title=None, save_dir=None,
                        show=True, xlabel='', ylabel='sigma'):
    for y_dict_point in y_dict_point_list:
        for name, y in y_dict_point.items():
            plt.scatter(x, y, s=1, label=name)
    for y_line, name in y_line_list:
        plt.plot(x, [y_line for _ in range(len(x))], label=name)
    if title:
        plt.title(title)
    plt.xlabel(xlabel + text_label)
    plt.ylabel(ylabel)
    plt.legend()
    if save_dir:
        save_name = (title if title else '') + '-'.join(y_dict_point_list[0].keys()) + '.png'
        save_name = save_name.replace('/', '|')
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    plt.close('all')
