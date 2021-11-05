import torch.utils.data

from loader import CsiDataloader
from model import BaseNetModel
from model import Tee
from utils.common import TestMethod
from utils.model import *


class PathEstBaseModel(BaseNetModel):

    def __init__(self, csiDataloader: CsiDataloader, test_method:TestMethod):
        super().__init__(csiDataloader)
        self.test_method = test_method


class PathEstDnn(PathEstBaseModel):

    def __init__(self, csiDataloader: CsiDataloader, add_var=True, use_true_var=False, dnn_list=None, extra=''):
        super().__init__(csiDataloader, TestMethod.one_row)
        if dnn_list is None:
            dnn_list = [csiDataloader.n_r * 4]
            start = csiDataloader.n_r * 4
            while start >= 32:
                dnn_list.append(start)
                start //= 2
        self.dnn_list = dnn_list
        self.add_var = add_var
        self.use_true_var = use_true_var
        self.extra = extra
        self.n_sc = self.csiDataloader.n_sc
        self.n_r = self.csiDataloader.n_r

        self.name = self.__str__()

        fc = [2 * self.csiDataloader.n_r + (1 if self.add_var else 0), ] + list(dnn_list) + [1, ]
        self.fc = []
        for i, j in zip(fc[:-1], fc[1:]):
            self.fc.append(nn.Linear(i, j))
            self.fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.fc)
        # self.sigmoid = nn.Sigmoid()

    def __str__(self):
        name = '{}-{}_r{}t{}K{}_dn{}-var{}_tvar{}{}'.format(self.get_dataset_name(), self.__class__.__name__, self.n_r,
                                                            self.n_t, self.n_sc, '-'.join(map(str, self.dnn_list)),
                                                            self.add_var, self.use_true_var, self.extra)
        return name

    def basename(self):
        return 'pathest'

    def get_short_name(self):
        return 'DnnPathEst'

    def forward(self, g: torch.Tensor, idx_row, g_row, true_var, est_var):
        var = true_var if self.use_true_var else est_var
        if self.add_var:
            g_row = torch.cat((var.reshape(-1, 1), g_row), dim=1)
        est_y = self.fc(g_row.view(g_row.size(0), -1))
        # est_y = self.sigmoid(est_y)
        est_y = torch.squeeze(est_y)
        return est_y,


class PathEstDnnLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bceLoss = torch.nn.BCELoss()

    def forward(self, right_y, est_y):
        return self.bceLoss(est_y, right_y)


class PathEstDnnTee(Tee):

    def __init__(self, items):
        super().__init__(items)
        self.g, self.idx_row, self.g_row, self.right_y, self.right_path, self.true_var, self.est_var = items
        self.est_y = None

    def get_model_input(self):
        return self.g, self.idx_row, self.g_row, self.true_var, self.est_var

    def set_model_output(self, outputs):
        self.est_y, = outputs

    def get_loss_input(self):
        return self.right_y, self.est_y


class PathEstCnn(PathEstBaseModel):

    def __init__(self, csiDataloader: CsiDataloader, add_var=True, use_true_var=False, cnn_count=4, cnn_channel=32,
                 dnn_list=(2000, 200, 50),
                 extra=''):
        super().__init__(csiDataloader, TestMethod.dft_diff)
        self.dnn_list = dnn_list
        self.add_var = add_var
        self.use_true_var = use_true_var
        self.extra = extra
        self.n_sc = self.csiDataloader.n_sc
        self.n_r = self.csiDataloader.n_r
        self.cnn_count = cnn_count
        self.cnn_channel = cnn_channel
        self.kernel_size = (3, 3)

        padding = ((self.kernel_size[0] - 1) // 2, (self.kernel_size[1] - 1) // 2)
        self.first_conv = ConvReluBlock(3 if self.add_var else 2, out_channel=cnn_channel, kernel_size=self.kernel_size,
                                        padding=padding, use_2dim=True)
        self.conv_bn_relu_seq = [ConvBnReluBlock(cnn_channel, cnn_channel, self.kernel_size, padding, True) for _ in
                                 range(cnn_count)]
        self.conv_bn_relu_seq = nn.Sequential(*self.conv_bn_relu_seq)
        self.back_conv = nn.Conv2d(self.cnn_channel, 1, kernel_size=self.kernel_size, padding=padding)

        fc = [self.n_sc * self.n_r] + list(dnn_list) + [1, ]
        self.fc = []
        for i, j in zip(fc[:-1], fc[1:]):
            self.fc.append(nn.Linear(i, j))
            self.fc.append(nn.Sigmoid())
        self.fc = nn.Sequential(*self.fc)

    def __str__(self):
        name = '{}-{}_r{}t{}K{}_c{}ke{}dn{}-var{}_tvar{}{}'.format(self.get_dataset_name(), self.__class__.__name__,
                                                                   self.n_r, self.n_t, self.n_sc, self.cnn_count,
                                                                   self.cnn_channel, '-'.join(map(str, self.dnn_list)),
                                                                   self.add_var, self.use_true_var, self.extra)
        return name

    def basename(self):
        return 'pathest'

    def get_short_name(self):
        return 'CnnPathEst'

    def forward(self, g_diff, idx_row, true_var, est_var):
        var = true_var if self.use_true_var else est_var
        if self.add_var:
            var_map = var.reshape(-1, 1, 1, 1).repeat(1, self.n_sc, self.n_r, 1)
            g_diff = torch.cat([g_diff, var_map], dim=-1)
        g_diff = g_diff.permute(0, 3, 1, 2)
        x = self.first_conv(g_diff)
        x = self.conv_bn_relu_seq(x)
        x = self.back_conv(x)
        est_y = self.fc(x.view(x.size(0), -1))
        est_y = torch.squeeze(est_y)
        return est_y,


class PathEstCnnLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bceLoss = torch.nn.BCELoss()

    def forward(self, right_y, est_y):
        return self.bceLoss(est_y, right_y)


class PathEstCnnTee(Tee):

    def __init__(self, items):
        super().__init__(items)
        self.g_diff, self.idx_row, self.right_y, self.right_path, self.true_var, self.est_var = items
        self.est_y = None

    def get_model_input(self):
        return self.g_diff, self.idx_row, self.true_var, self.est_var

    def set_model_output(self, outputs):
        self.est_y, = outputs

    def get_loss_input(self):
        return self.right_y, self.est_y
