import matplotlib.pyplot as plt
import os

plt.rcParams['axes.unicode_minus'] = False
save_dir = 'result/'


def draw_line(x, y_dict: dict, title=None, filter_func=None, save=True, show=True):
    if filter_func is None:
        filter_func = lambda n: True
    for name, y in y_dict.items():
        y = list(map(lambda n: n if filter_func(n) else None, y))
        plt.plot(x, y, label=name)
    if title:
        plt.title(title)
    plt.xlabel('snr(db)')
    plt.ylabel('nmse(db)')
    plt.legend()
    if save:
        save_name = (title if title else '') + '-'.join(y_dict.keys()) + '.png'
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, format='png')
    if show:
        plt.show()
    plt.close('all')
