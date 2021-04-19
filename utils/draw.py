import matplotlib.pyplot as plt


def draw_line(x, y_dict:dict):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for name, y in y_dict.items():
        plt.plot(x, y, label=name)
    plt.legend()
    plt.show()
