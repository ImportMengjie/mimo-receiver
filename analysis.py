import json
import os

from config import config
from utils import draw_line


def analysis_loss_nmse():
    for name in os.listdir(config.RESULT):
        if name.endswith('.json') and name.startswith('loss_nmse'):
            with open(os.path.join(config.RESULT, name)) as f:
                loss_data = json.load(f)
            shortname = loss_data['shortname']
            basename = loss_data.get('basename')
            loss_list = loss_data['loss']
            x = [i for i in range(0, len(loss_list))]
            draw_line(x, {shortname: loss_list}, title='loss', xlabel='iteration', ylabel='nmse(db)',
                      save_dir=config.RESULT_IMG, diff_line_markers=True)


if __name__ == '__main__':
    analysis_loss_nmse()
