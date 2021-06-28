import torch.nn as nn
from loader import CsiDataloader


class BaseNetModel(nn.Module):

    def __init__(self, csiDataloader: CsiDataloader):
        super().__init__()
        self.csiDataloader = csiDataloader
        self.name = BaseNetModel.__class__.__name__

    def get_dataset_name(self):
        return self.csiDataloader.__str__()

    def get_train_state(self):
        return {}

    def basename(self):
        return 'base'
