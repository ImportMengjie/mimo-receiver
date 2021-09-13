import json
import logging
import os
import time
from dataclasses import dataclass

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from loader import CsiDataloader, BaseDataset
from loader import DataType
from loader import DenoisingNetDataset
from loader import DetectionNetDataset
from loader import InterpolationNetDataset
from model import BaseNetModel
from model import DenoisingNetLoss
from model import CBDNetBaseModel
from model import DenoisingNetTee
from model import DetectionNetLoss
from model import DetectionNetModel
from model import DetectionNetTee
from model import InterpolationNetLoss
from model import CBDNetSFModel
from model import InterpolationNetTee
from model import Tee
from utils import AvgLoss, print_parameter_number
import utils.config as config


# from torchsummary import summary


@dataclass()
class TrainParam:
    lr: float = 0.001
    epochs: int = 10000
    momentum: float = 0.9
    batch_size: int = 64
    use_scheduler: bool = True
    stop_when_test_loss_down_epoch_count = 20
    log_loss_per_epochs: int = 10


class Train:
    save_dir = 'result/'
    save_per_epoch = 5

    def __init__(self, param: TrainParam, dataset: BaseDataset, model: BaseNetModel, criterion: Module,
                 teeClass: Tee.__class__, test_dataset: BaseDataset):
        self.param = param
        self.model = model
        self.criterion = criterion
        self.teeClass = teeClass
        self.losses = []
        if config.USE_GPU:
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            # dataset.cuda()
            # test_dataset.cuda()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, param.batch_size, True)
        self.test_dataset = test_dataset
        self.test_dataloader = None
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, param.batch_size)

    def get_save_path(self):
        return Train.get_save_path_from_model(self.model)

    def reset_current_epoch(self):
        if os.path.exists(self.get_save_path()):
            model_info = torch.load(self.get_save_path())
            if 'epoch' in model_info:
                model_info['epoch'] = 0
                torch.save(model_info, self.get_save_path())

    def train(self, save=True, reload=True, ext_log: str = '', loss_data_func=None):
        logging.warning('start train:{}'.format(str(self.model)))
        self.losses.clear()
        current_epoch = 0
        test_loss = []
        test_loss_data = None
        save_loss_data_path = None
        if loss_data_func:
            test_loss_data = [0]
            save_loss_data_path = os.path.join(config.RESULT,
                                               "loss_nmse_{}_{}.json".format(self.model.__str__(), int(time.time())))
        Train.save_per_epoch = self.param.stop_when_test_loss_down_epoch_count

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
        avg_loss = AvgLoss()
        test_avg_loss = AvgLoss()
        while True:
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
            if loss_data_func:
                test_loss_data.append(loss_data_func(self))
                with open(save_loss_data_path, 'w') as f:
                    json.dump({'shortname': self.model.get_short_name(), 'basename': self.model.basename(),
                               'loss': test_loss_data}, f)
            if current_epoch % self.param.log_loss_per_epochs == 0:
                logging.info(
                    'epoch:{} avg_loss:{};{}'.format(current_epoch, self.losses[-1], ext_log))
            if self.test_dataloader and current_epoch % self.param.stop_when_test_loss_down_epoch_count == 0 and current_epoch >= self.param.epochs:
                for items in self.test_dataloader:
                    tee = self.teeClass(items)
                    tee.set_model_output(self.model.eval()(*tee.get_model_input()))
                    loss = self.criterion(*tee.get_loss_input())
                    test_avg_loss.add(loss.item())
                test_loss.append(test_avg_loss.avg)
                test_avg_loss.reset()
                logging.warning('test loss:{} in epoch:{}'.format(test_loss[-1], current_epoch))
                if len(test_loss) > 1 and test_loss[-1] > test_loss[-2]:
                    logging.error('test loss down [-2]{}, [-1]{}'.format(test_loss[-2], test_loss[-1]))
                    break
                self.model.train()
            if self.param.use_scheduler:
                scheduler.step()
            if save and (current_epoch % Train.save_per_epoch == 0):
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
        return os.path.join(Train.save_dir, model.basename(), '{}.pth.tar'.format(str(model)))


def load_model_from_file(model, use_gpu: bool):
    save_model_path = Train.get_save_path_from_model(model)
    if os.path.exists(save_model_path):
        model_info = torch.load(save_model_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_info['state_dict'])
        model = model.double().eval()
        if use_gpu:
            model = model.cuda()
    else:
        logging.warning('unable load {} model'.format(save_model_path))
    return model


def train_denoising_net(data_path: str, snr_range: list, noise_num=4, noise_channel=32, denoising_num=6,
                        denoising_channel=64, only_train_noise_level=False, use_true_sigma=False, fix_noise=False,
                        extra=''):
    assert not (only_train_noise_level and use_true_sigma)
    assert not (only_train_noise_level and fix_noise)
    csi_dataloader = CsiDataloader(data_path, factor=1, train_data_radio=0.9)
    dataset = DenoisingNetDataset(csi_dataloader, DataType.train, snr_range)
    test_dataset = DenoisingNetDataset(csi_dataloader, DataType.test, snr_range)

    model = CBDNetBaseModel(csi_dataloader, noise_level_conv_num=noise_num, noise_channel_num=noise_channel,
                            denosing_conv_num=denoising_num, denosing_channel_num=denoising_channel,
                            use_true_sigma=use_true_sigma, only_return_noise_level=only_train_noise_level, extra=extra)
    print_parameter_number(model)
    if fix_noise:
        for p in model.noise_level.parameters():
            p.requires_grad = False
    criterion = DenoisingNetLoss(only_train_noise_level=only_train_noise_level)
    param = TrainParam()
    param.stop_when_test_loss_down_epoch_count = 10
    param.epochs = 50
    param.lr = 0.001
    param.batch_size = 100
    param.log_loss_per_epochs = 1

    train = Train(param, dataset, model, criterion, DenoisingNetTee, test_dataset)
    train.train()


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
