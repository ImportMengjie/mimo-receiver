import logging
import os
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
from model import DenoisingNetModel
from model import DenoisingNetTee
from model import DetectionNetLoss
from model import DetectionNetModel
from model import DetectionNetTee
from model import InterpolationNetLoss
from model import InterpolationNetModel
from model import InterpolationNetTee
from model import Tee
from utils import AvgLoss


# from torchsummary import summary


@dataclass()
class TrainParam:
    lr: float = 0.001
    epochs: int = 10000
    momentum: float = 0.9
    use_gpu: bool = True
    batch_size: int = 64
    use_scheduler: bool = True
    stop_when_test_loss_down_epoch_count = 20


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
        if self.param.use_gpu and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = self.criterion.cuda()
            dataset.cuda()
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, param.batch_size, True)
        self.test_dataset = test_dataset
        self.test_dataloader = None
        if self.test_dataset:
            self.test_dataloader = DataLoader(self.test_dataset, len(self.test_dataset))

    def get_save_path(self):
        return Train.get_save_path_from_model(self.model)

    def reset_current_epoch(self):
        if os.path.exists(self.get_save_path()):
            model_info = torch.load(self.get_save_path())
            if 'epoch' in model_info:
                model_info['epoch'] = 0
                torch.save(model_info, self.get_save_path())

    def train(self, save=True, reload=True, ext_log: str = ''):
        self.losses.clear()
        current_epoch = 0
        test_loss = []

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
        while True:
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
            if current_epoch % 10 == 0:
                logging.info(
                    'epoch:{} avg_loss:{};{}'.format(current_epoch, self.losses[-1], ext_log))
            if self.test_dataloader and current_epoch % self.param.stop_when_test_loss_down_epoch_count == 0 and current_epoch >= self.param.epochs:
                items = list(self.test_dataloader)[0]
                tee = self.teeClass(items)
                tee.set_model_output(self.model.eval()(*tee.get_model_input()))
                loss = self.criterion(*tee.get_loss_input())
                test_loss.append(loss.item())
                logging.warning('test loss:{} in epoch:{}'.format(loss, current_epoch))
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
        return os.path.join(Train.save_dir, '{}.pth.tar'.format(str(model)))


def train_denoising_net(data_path: str, snr_range: list, ):
    csi_dataloader = CsiDataloader(data_path)
    dataset = DenoisingNetDataset(csi_dataloader, DataType.train, snr_range)
    test_dataset = DenoisingNetDataset(csi_dataloader, DataType.test, snr_range)

    model = DenoisingNetModel(csi_dataloader.n_r, csi_dataloader.n_t)
    criterion = DenoisingNetLoss()
    param = TrainParam()
    param.loss_not_down_stop_count = 10
    param.epochs = 10
    param.lr = 0.001

    train = Train(param, dataset, model, criterion, DenoisingNetTee, test_dataset)
    train.train()


def train_interpolation_net(data_path: str, snr_range: list, pilot_count: int):
    csi_dataloader = CsiDataloader(data_path)
    dataset = InterpolationNetDataset(csi_dataloader, DataType.train, snr_range, pilot_count)
    test_dataset = InterpolationNetDataset(csi_dataloader, DataType.test, snr_range, pilot_count)

    model = InterpolationNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_sc, pilot_count)
    criterion = InterpolationNetLoss()
    param = TrainParam()

    train = Train(param, dataset, model, criterion, InterpolationNetTee, test_dataset)
    train.train()


def train_detection_net(data_path: str, training_snr: list, modulation='qpsk', save=True, reload=True, retrain=False):
    refinements = [.5, .1, .01]
    lr = 1e-3

    def get_nmse(model: DetectionNetModel, dataset: DetectionNetDataset):
        nmses = {}
        for snr in range(0, 30, 2):
            n, var = dataset.csiDataloader.noise_snr_range(dataset.hx, [snr, snr + 1], True)
            y = dataset.hx + dataset.n
            A = dataset.h.conj().transpose(-1, -2) @ dataset.h + var * torch.eye(dataset.csiDataloader.n_t,
                                                                                 dataset.csiDataloader.n_t)
            b = dataset.h.conj().transpose(-1, -2) @ y
            x = dataset.x

            # x = dataset.csiDataloader.get_x(dataset.dataType, dataset.modulation)
            # x = torch.cat((x.real, x.imag), 2)

            b = torch.cat((b.real, b.imag), 2)
            A_left = torch.cat((A.real, A.imag), 2)
            A_right = torch.cat((-A.imag, A.real), 2)
            A = torch.cat((A_left, A_right), 3)

            x_hat, = model(A, b)
            nmse = (10 * torch.log10((((x - x_hat) ** 2).sum(-1).sum(-1) / (x ** 2).sum(-1).sum(-1)).mean())).item()
            nmses[snr] = nmse
        return nmses

    csi_dataloader = CsiDataloader(data_path, factor=10000)
    model = DetectionNetModel(csi_dataloader.n_r, csi_dataloader.n_t, csi_dataloader.n_r * 2, True,
                              modulation=modulation)
    if retrain and os.path.exists(Train.get_save_path_from_model(model)):
        model_info = torch.load(Train.get_save_path_from_model(model))
        model_info['snr'] = training_snr[0]
        model_info['epoch'] = 0
        model.set_training_layer(32, False)
        model_info['train_state'] = model.get_train_state()
        torch.save(model_info, Train.get_save_path_from_model(model))
    criterion = DetectionNetLoss()
    param = TrainParam()
    param.batch_size = 100
    param.use_scheduler = False
    training_snr = sorted(training_snr, reverse=True)
    if reload and os.path.exists(Train.get_save_path_from_model(model)):
        model_info = torch.load(Train.get_save_path_from_model(model))
        if 'snr' in model_info and model_info['snr'] in training_snr:
            logging.warning('snr list:{}, start snr:{}'.format(training_snr, model_info['snr']))
            training_snr = training_snr[training_snr.index(model_info['snr']):]

    test_dataset = DetectionNetDataset(csi_dataloader, DataType.test, [5, 40], modulation)

    def train_fixed_snr(snr_: int):
        dataset = DetectionNetDataset(csi_dataloader, DataType.train, [snr_, snr_ + 1], modulation)
        train = Train(param, dataset, model, criterion, DetectionNetTee, test_dataset)
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
                train.param.loss_not_down_stop_count = 50
                train.param.epochs = 10
                train.param.lr = 0.001
                model.set_training_layer(layer_num, True)
                train.train(save=save, reload=reload,
                            ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))
                train.reset_current_epoch()

            over_fix_forward = False
            logging.info('Fine tune layer:{}'.format(layer_num))
            train.param.loss_not_down_stop_count = 50
            train.param.epochs = 10

            learn_rate = train.param.lr
            for factor in refinements:
                train.param.lr = learn_rate * factor
                model.set_training_layer(layer_num, False)
                train.train(save=save, reload=reload,
                            ext_log='snr:{},model:{},lr:{}'.format(snr, model.get_train_state_str(), param.lr))
                train.reset_current_epoch()
        return dataset

    for snr in training_snr:
        param.lr = lr
        dataset = train_fixed_snr(snr)
        # logging.warning('NMSE Loss:{}'.format(get_nmse(model, dataset)))
        if save and os.path.exists(Train.get_save_path_from_model(model)):
            model_infos = torch.load(Train.get_save_path_from_model(model))
            model_infos.pop('train_state')
            torch.save(model_infos, Train.get_save_path_from_model(model))


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    # train_denoising_net('data/normal_3gpp_16_16_64_5_5.mat', [100, 201])
    # train_interpolation_net('data/3gpp_16_16_64_5_5.mat', [50, 51], 4)
    # train_detection_net('data/gaussian_16_16_1_100.mat', [60, 50, 20])
    train_detection_net('data/gaussian_16_16_1_1.mat', [30, 20, 15, 10], retrain=True, modulation='qpsk')
