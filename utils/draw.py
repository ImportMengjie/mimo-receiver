import matplotlib.pyplot as plt
import os

plt.rcParams['axes.unicode_minus'] = False


def draw_line(x, y_dict: dict, title=None, filter_func=None, save_dir=None, show=True, xlabel='snr(db)', ylabel='nmse(db)'):
    if filter_func is None:
        filter_func = lambda n: True
    for name, y in y_dict.items():
        y = list(map(lambda n: n if filter_func(n) else None, y))
        plt.plot(x, y, label=name)
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
