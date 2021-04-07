import logging
import os
from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from loader import CsiDataloader
from loader import DataType
from loader import DenoisingNetDataset
from model import DenoisingNetLoss
from model import DenoisingNetModel
from model import DenoisingNetTee
from model import Tee
from utils import AvgLoss


@dataclass()
class TrainParam:
    lr: int = 0.0000001
    epochs: int = 100


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
        save_path = os.path.join(Train.save_dir, '{}.pth.tar'.format(str(model)))
        current_epoch = 0
        if reload and os.path.exists(save_path):
            model_info = torch.load(save_path)
            model.load_state_dict(model_info['state_dict'])
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(model_info['optimizer'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.param.epochs)
            scheduler.load_state_dict(model_info['scheduler'])
            current_epoch = model_info['epoch']
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param.lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.param.epochs)
        self.model.train()
        self.model.double()
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
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(Train.save_dir, '{}.pth.tar'.format(str(model))))


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    csiDataloader = CsiDataloader('data/h_16_16_64_1.mat')
    dataset = DenoisingNetDataset(csiDataloader, DataType.train, [50, 51])
    dataloader = torch.utils.data.DataLoader(dataset, 10, True)

    model = DenoisingNetModel(csiDataloader.n_r, csiDataloader.n_t)
    criterion = DenoisingNetLoss()
    param = TrainParam()

    train = Train(param, dataloader, model, criterion, DenoisingNetTee)
    train.train()
