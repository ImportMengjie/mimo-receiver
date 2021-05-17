import logging
import os
from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
# from torchsummary import summary

from loader import CsiDataloader, BaseDataset
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
from model import BaseNetModel


@dataclass()
class TrainParam:
    lr: float = 0.0000001
    epochs: int = 10000
    momentum: float = 0.9
    loss_not_down_stop_count: int = 10
    use_gpu: bool = True
    batch_size: int = 64
    use_scheduler: bool = True


class Train:
    save_dir = 'result/'
    save_per_epoch = 5

    def __init__(self, param: TrainParam, dataset: BaseDataset, model: BaseNetModel, criterion: Module,
                 teeClass: Tee.__class__):
        self.param = param
        self.model = model
        self.criterion = criterion
        self.teeClass = teeClass
        self.losses = []
        if self.param.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            dataset.cuda()
        self.dataloader = DataLoader(dataset, param.batch_size, True)

    def get_save_path(self):
        return Train.get_save_path_from_model(self.model)

    def reset_current_epoch(self):
        if os.path.exists(self.get_save_path()):
            model_info = torch.load(self.get_save_path())
            if 'epoch' in model_info:
                model_info['epoch'] = 0
                torch.save(model_info, self.get_save_path())

    def train(self, save=True, reload=True, ext_log: str = '', ):
        self.losses.clear()
        current_epoch = 0

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.param.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.param.epochs)
        model_info = None
        if reload and os.path.exists(self.get_save_path()):
            model_info = torch.load(self.get_save_path())
            if 'state_dict' in model_info:
                self.model.load_state_dict(model_info['state_dict'])
                # optimizer.load_state_dict(model_info['optimizer'])
                scheduler.load_state_dict(model_info['scheduler'])
                current_epoch = model_info['epoch']

        self.model.train()
        self.model.double()
        # logging.info('model:')
        # summary(self.model)
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
            if len(self.losses) > 2:
                if current_epoch > self.param.epochs and self.losses[-1] > self.losses[-2]:
                    loss_down_count -= 1
                elif current_epoch > self.param.epochs and loss_down_count < self.param.loss_not_down_stop_count and \
                        self.losses[-1] < self.losses[-2] and loss_down_count % 2 == 1:
                    loss_down_count += 1
            avg_loss.reset()
            logging.info(
                'epoch:{} avg_loss:{} countdown:{};{}'.format(current_epoch, self.losses[-1], loss_down_count, ext_log))
            if self.param.use_scheduler:
                scheduler.step()
            if save and (current_epoch % Train.save_per_epoch == 0 or current_epoch + 1 == self.param.epochs):
                # save model
                if model_info is None:
                    model_info = {}
                model_info['epoch'] = current_epoch + 1
                model_info['state_dict'] = self.model.state_dict()
                model_info['optimizer'] = optimizer.state_dict()
                model_info['scheduler'] = scheduler.state_dict()
                model_info['train_state'] = self.model.get_train_state()
                torch.save(model_info, self.get_save_path())
            current_epoch += 1

    @staticmethod
    def get_save_path_from_model(model: BaseNetModel):
        return os.path.join(Train.save_dir, '{}.pth.tar'.format(str(model)))


def train_denoising_net(data_path: str, snr_range: list, ):
    csi_dataloader = CsiDataloader(data_path)
    dataset = DenoisingNetDataset(csi_dataloader, DataType.train, snr_range)

    model = DenoisingNetModel(csi_dataloader.n_r, csi_dataloader.n_t)
    criterion = DenoisingNetLoss()
    param = TrainParam()
    param.loss_not_down_stop_count = 100
    param.epochs = 10

    train = Train(param, dataset, model, criterion, DenoisingNetTee)
    train.train()


def train_interpolation_net(data_path: str, snr_range: list, pilot_count: int):
    csi_dataloader = CsiDataloader(data_path)
    dataset = InterpolationNetDataset(csi_dataloader, DataType.train, snr_range, pilot_count)

    model = InterpolationNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_sc, pilot_count)
    criterion = InterpolationNetLoss()
    param = TrainParam()

    train = Train(param, dataset, model, criterion, InterpolationNetTee)
    train.train()


def train_detection_net(data_path: str, training_snr: list, modulation='qpsk', save=True, reload=True):
    csi_dataloader = CsiDataloader(data_path)
    model = DetectionNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_r, True, modulation=modulation)
    criterion = DetectionNetLoss()
    param = TrainParam()
    param.lr = 0.001
    param.epochs = 5000
    param.batch_size = csi_dataloader.n_sc
    training_snr = sorted(training_snr, reverse=True)
    if reload and os.path.exists(Train.get_save_path_from_model(model)):
        model_info = torch.load(Train.get_save_path_from_model(model))
        if 'snr' in model_info and model_info['snr'] in training_snr:
            logging.warning('snr list:{}, start snr:{}'.format(training_snr, model_info['snr']))
            training_snr = training_snr[training_snr.index(model_info['snr']):]

    def train_fixed_snr(snr_: int):
        dataset = DetectionNetDataset(csi_dataloader, DataType.train, [snr_, snr_ + 1], modulation)
        train = Train(param, dataset, model, criterion, DetectionNetTee)
        model_infos = None
        current_train_layer = 1
        over_fix_forward = False
        if reload and os.path.exists(train.get_save_path()):
            model_infos = torch.load(train.get_save_path())
            if 'train_state' in model_infos:
                current_train_layer = model_infos['train_state']['train_layer']
                over_fix_forward = not model_infos['train_state']['fix_forward']
                logging.warning('load train state:{}'.format(model_infos['train_state']))
        if save:
            if model_infos is None:
                model_infos = {}
            model_infos['snr'] = snr_
            torch.save(model_infos, train.get_save_path())

        for layer_num in range(current_train_layer, model.layer_nums + 1):
            if not over_fix_forward:
                logging.info('training layer:{}'.format(layer_num))
                train.param.loss_not_down_stop_count = 10
                train.param.epochs = 1000
                train.param.lr = 0.01
                train.param.use_scheduler = True
                model.set_training_layer(layer_num, True)
                train.train(save=save, reload=reload,
                            ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))
                train.reset_current_epoch()

            over_fix_forward = False
            logging.info('Fine tune layer:{}'.format(layer_num))
            train.param.loss_not_down_stop_count = 10
            train.param.lr = 0.01 * 0.5 ** layer_num
            train.param.epochs = 1000
            train.param.use_scheduler = True
            model.set_training_layer(layer_num, False)
            train.train(save=save, reload=reload, ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))
            train.reset_current_epoch()

    for snr in training_snr:
        train_fixed_snr(snr)


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    # train_denoising_net('data/h_16_16_64_1.mat', [50, 51])
    # train_interpolation_net('data/h_16_16_64_1.mat', [50, 51], 4)
    # train_detection_net('data/h_16_16_64_1.mat', [200, 150, 100, 50, 20])
    train_detection_net('data/h_16_16_64_5.mat', [200])
