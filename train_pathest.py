from loader.PathEstNetDataset import PathEstCnnNetDataset
from model.PathEstNetModel import PathEstCnn, PathEstDnnLoss, PathEstCnnTee
from train import *
from model import PathEstDnn, PathEstCnnLoss, PathEstDnnTee
from loader import PathEstDnnNetDataset


def train_pathest_dnn_net(data_path: str, snr_range: list, path_range: list, add_var=True, use_true_var=False,
                          dnn_list=None, fix_path=None, extra='', reload=True):
    csi_dataloader = CsiDataloader(data_path, train_data_radio=0.9)
    dataset = PathEstDnnNetDataset(csiDataloader=csi_dataloader, dataType=DataType.train, snr_range=snr_range,
                                   path_range=path_range, fix_path=fix_path)
    test_dataset = PathEstDnnNetDataset(csiDataloader=csi_dataloader, dataType=DataType.test, snr_range=snr_range,
                                        path_range=path_range, fix_path=fix_path)
    model = PathEstDnn(csiDataloader=csi_dataloader, add_var=add_var, use_true_var=use_true_var, dnn_list=dnn_list,
                       extra=extra)
    criterion = PathEstDnnLoss()
    param = TrainParam()
    param.stop_when_test_loss_down_epoch_count = 5
    param.epochs = 20
    param.lr = 0.001
    param.batch_size = 100
    param.log_loss_per_epochs = 1

    train = Train(param, dataset, model, criterion, PathEstDnnTee, test_dataset)
    train.train(reload=reload, weight_decay=0.)


def train_pathest_cnn_net(data_path: str, snr_range: list, path_range: list, add_var, use_true_var,
                          cnn_count, cnn_channel, dnn_list, fix_path=None, extra='', reload=True):
    csi_dataloader = CsiDataloader(data_path, train_data_radio=0.9)
    dataset = PathEstCnnNetDataset(csiDataloader=csi_dataloader, dataType=DataType.train, snr_range=snr_range,
                                   path_range=path_range, fix_path=fix_path)
    test_dataset = PathEstCnnNetDataset(csiDataloader=csi_dataloader, dataType=DataType.test, snr_range=snr_range,
                                        path_range=path_range, fix_path=fix_path)
    model = PathEstCnn(csiDataloader=csi_dataloader, add_var=add_var, use_true_var=use_true_var, cnn_count=cnn_count,
                       cnn_channel=cnn_channel, dnn_list=dnn_list, extra=extra)
    criterion = PathEstCnnLoss()
    param = TrainParam()
    param.stop_when_test_loss_down_epoch_count = 5
    param.epochs = 20
    param.lr = 0.001
    param.batch_size = 100
    param.log_loss_per_epochs = 1

    train = Train(param, dataset, model, criterion, PathEstCnnTee, test_dataset)
    train.train(reload=reload, weight_decay=0.)


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    # train_pathest_dnn_net(data_path='data/imt_2020_64_32_64_400.mat', snr_range=[1, 15], path_range=[1, 20],
    #                       add_var=True,
    #                       use_true_var=False, dnn_list=[256, 256, 128, 128, 64, 32, ], fix_path=None, extra='',
    #                       reload=False)
    train_pathest_cnn_net(data_path='data/imt_2020_64_32_64_400.mat', snr_range=[1, 15], path_range=[1, 20],
                          add_var=True, use_true_var=False, dnn_list=[2000, 200, 20, ], cnn_count=1,
                          cnn_channel=32, fix_path=None, extra='', reload=False)
    # train_pathest_net(data_path='data/imt_2020_64_32_64_400.mat', snr_range=[0, 11], path_range=[3, 16], add_var=True,
    #                   use_true_var=False, dnn_list=None, fix_path=None, extra='',
    #                   retrain=True)
    pass
