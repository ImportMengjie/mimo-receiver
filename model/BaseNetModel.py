import torch.nn as nn


class BaseNetModel(nn.Module):

    def get_train_state(self):
        return {}
