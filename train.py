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

    def train(self, save=True, reload=True):
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
        for epoch in range(current_epoch, self.param.epochs):
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
            avg_loss.reset()
            logging.info('epoch:{} avg_loss:{}'.format(epoch, self.losses[-1]))
            scheduler.step()

            if save and (epoch % Train.save_per_epoch == 0 or epoch + 1 == self.param.epochs):
                # save model
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(Train.save_dir, '{}.pth.tar'.format(str(self.model))))


def train_denoising_net(data_path: str, snr_range: list, ):
    csiDataloader = CsiDataloader(data_path)
    dataset = DenoisingNetDataset(csiDataloader, DataType.train, snr_range)
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = DenoisingNetModel(csiDataloader.n_r, csiDataloader.n_t)
    criterion = DenoisingNetLoss()
    param = TrainParam()

    train = Train(param, dataloader, model, criterion, DenoisingNetTee)
    train.train()


def train_interpolation_net(data_path: str, snr_range: list, pilot_count: int):
    csiDataloader = CsiDataloader(data_path)
    dataset = InterpolationNetDataset(csiDataloader, DataType.train, snr_range, pilot_count)
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = InterpolationNetModel(csiDataloader.n_r, csiDataloader.n_t)
    criterion = InterpolationNetLoss()
    param = TrainParam()

    train = Train(param, dataloader, model, criterion, InterpolationNetTee)
    train.train()


def train_detection_net(data_path: str, training_snr: list, modulation='qpsk'):
    csiDataloader = CsiDataloader(data_path)
    model = DetectionNetModel(csiDataloader.n_r, csiDataloader.n_t, 10, True, modulation=modulation)
    criterion = DetectionNetLoss()
    param = TrainParam()
    param.epochs = 100
    training_snr = sorted(training_snr, reverse=True)

    def train_fixed_snr(snr_: int):
        dataset = DetectionNetDataset(csiDataloader, DataType.train, [snr_, snr_ + 1], modulation)
        dataloader = torch.utils.data.DataLoader(dataset, 10, True)
        train = Train(param, dataloader, model, criterion, DetectionNetTee)
        for layer_num in range(1, model.layer_nums+1):
            logging.info('training layer:{}'.format(layer_num))
            model.set_training_layer(layer_num, True)
            train.train(reload=False)
            logging.info('Fine tune layer:{}'.format(layer_num))
            model.set_training_layer(layer_num, False)
            train.train(reload=False)

    for snr in training_snr:
        train_fixed_snr(snr)


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    # train_denoising_net('data/h_16_16_64_1.mat', [50, 51])
    # train_interpolation_net('data/h_16_16_64_1.mat', [50, 51], 4)
    train_detection_net('data/h_16_16_64_1.mat', [170, 160])
