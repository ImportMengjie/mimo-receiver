import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False


def draw_line(x, y_dict: dict, filter_func=None):
    if filter_func is None:
        filter_func = lambda n: True
    for name, y in y_dict.items():
        y = list(map(lambda n: n if filter_func(n) else None, y))
        plt.plot(x, y, label=name)
    plt.legend()
    plt.show()
