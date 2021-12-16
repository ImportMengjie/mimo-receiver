# from utils import draw_line
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))

# from utils import draw_line
from draw import draw_line


def redraw(json_path: str):
    """
    data_json = {
        'title': title,
        'x': x,
        'y_dict': y_dict,
        'xlabel': xlabel,
        'ylable': ylabel,
        'style': diff_line_style,
        'markers': diff_line_markers
    }
    :param json_path:
    :return:
    """
    with open(json_path) as f:
        data_json = json.load(f)
    path, _ = os.path.splitext(json_path)
    img_path = path + '.png'
    draw_line(x=data_json['x'], y_dict=data_json['y_dict'], title=data_json['title'], save_path=img_path,
              xlabel=data_json['xlabel'], ylabel=data_json['ylabel'], diff_line_style=data_json['style'],
              diff_line_markers=data_json['markers'])


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    path_list = sys.argv[1:]
    for json_path in path_list:
        logging.info('redraw:{}'.format(json_path))
        redraw(json_path)
