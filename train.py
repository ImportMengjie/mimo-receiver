import logging
import os
from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchsummary import summary

from loader import CsiDataloader
from loader import DataType
from loader import DenoisingNetDataset
from loader import InterpolationNetDataset
from loader import DetectionNetDataset

from model import DenoisingNetLoss
from model import DenoisingNetModel
from model import DenoisingNetTee

from model import InterpolationNetModel
from model import InterpolationNetLoss
from model import InterpolationNetTee

from model import DetectionNetModel
from model import DetectionNetLoss
from model import DetectionNetTee

from utils import AvgLoss
from model import Tee


@dataclass()
class TrainParam:
    lr: float = 0.0000001
    epochs: int = 100
    momentum: float = 0.9
    loss_not_down_stop_count: int = 10
    use_gpu: bool = True


class Train:
    save_dir = 'result/'
    save_per_epoch = 5

    def __init__(self, param: TrainParam, dataloader: torch.utils.data.DataLoader, model: Module, criterion: Module,
                 teeClass: Tee.__class__):
        self.param = param
        self.dataloader = dataloader
        self.model = model
        self.criterion = criterion
        self.teeClass = teeClass
        self.losses = []
        if self.param.use_gpu and torch.cuda.is_available():
            self.model.cuda()
            self.criterion.cuda()

    def train(self, save=True, reload=True, ext_log: str = ''):
        self.losses.clear()
        save_path = os.path.join(Train.save_dir, '{}.pth.tar'.format(str(self.model)))
        current_epoch = 0

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.param.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.param.epochs)
        if reload and os.path.exists(save_path):
            model_info = torch.load(save_path)
            self.model.load_state_dict(model_info['state_dict'])
            optimizer.load_state_dict(model_info['optimizer'])
            scheduler.load_state_dict(model_info['scheduler'])
            current_epoch = model_info['epoch']

        self.model.train()
        self.model.double()
        logging.info('model:')
        summary(self.model)
        loss_down_count = self.param.loss_not_down_stop_count
        while current_epoch < self.param.epochs or loss_down_count > 0:
            avg_loss = AvgLoss()
            for items in self.dataloader:
                tee = self.teeClass(items)
                tee.set_model_output(self.model(*tee.get_model_input()))
                loss = self.criterion(*tee.get_loss_input())
                avg_loss.add(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.losses.append(avg_loss.avg)
            if current_epoch > self.param.epochs and self.losses[-1] > self.losses[-2]:
                loss_down_count -= 1
            elif current_epoch > self.param.epochs and loss_down_count < self.param.loss_not_down_stop_count and \
                    self.losses[-1] < self.losses[-2] and loss_down_count % 2 == 1:
                loss_down_count += 1
            avg_loss.reset()
            logging.info('epoch:{} avg_loss:{};{}'.format(current_epoch, self.losses[-1], ext_log))
            scheduler.step()
            if save and (current_epoch % Train.save_per_epoch == 0 or current_epoch + 1 == self.param.epochs):
                # save model
                torch.save({
                    'epoch': current_epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(Train.save_dir, '{}.pth.tar'.format(str(self.model))))
            current_epoch += 1


def train_denoising_net(data_path: str, snr_range: list, ):
    csi_dataloader = CsiDataloader(data_path)
    dataset = DenoisingNetDataset(csi_dataloader, DataType.train, snr_range)
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = DenoisingNetModel(csi_dataloader.n_r, csi_dataloader.n_t)
    criterion = DenoisingNetLoss()
    param = TrainParam()
    param.loss_not_down_stop_count = 100
    param.epochs = 10

    train = Train(param, dataloader, model, criterion, DenoisingNetTee)
    train.train()


def train_interpolation_net(data_path: str, snr_range: list, pilot_count: int):
    csi_dataloader = CsiDataloader(data_path)
    dataset = InterpolationNetDataset(csi_dataloader, DataType.train, snr_range, pilot_count)
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = InterpolationNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_sc, pilot_count)
    criterion = InterpolationNetLoss()
    param = TrainParam()

    train = Train(param, dataloader, model, criterion, InterpolationNetTee)
    train.train()


def train_detection_net(data_path: str, training_snr: list, modulation='qpsk'):
    csi_dataloader = CsiDataloader(data_path)
    model = DetectionNetModel(csi_dataloader.n_r, csi_dataloader.n_t, 10, True, modulation=modulation)
    criterion = DetectionNetLoss()
    param = TrainParam()
    param.lr = 0.01
    param.epochs = 5000
    training_snr = sorted(training_snr, reverse=True)

    def train_fixed_snr(snr_: int):
        dataset = DetectionNetDataset(csi_dataloader, DataType.train, [snr_, snr_ + 1], modulation)
        dataloader = torch.utils.data.DataLoader(dataset, 10, True)
        train = Train(param, dataloader, model, criterion, DetectionNetTee)
        for layer_num in range(1, model.layer_nums + 1):
            logging.info('training layer:{}'.format(layer_num))
            train.param.loss_not_down_stop_count = 10
            model.set_training_layer(layer_num, True)
            train.train(reload=False, ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))
            logging.info('Fine tune layer:{}'.format(layer_num))
            train.param.loss_not_down_stop_count = 100
            model.set_training_layer(layer_num, False)
            train.train(reload=False, ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))

    for snr in training_snr:
        train_fixed_snr(snr)


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    # train_denoising_net('data/h_16_16_64_1.mat', [50, 51])
    # train_interpolation_net('data/h_16_16_64_1.mat', [50, 51], 4)
    # train_detection_net('data/h_16_16_64_1.mat', [200, 150, 100, 50, 20])
    train_detection_net('data/h_16_16_64_1.mat', [200])
