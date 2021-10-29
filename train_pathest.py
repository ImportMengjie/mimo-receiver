from train import *
from model import DnnPathEst, PathEstNetLoss, PathEstNetTee
from loader import PathEstNetDataset


def train_pathest_net(data_path: str, snr_range: list, path_range: list, add_var=True, use_true_var=False,
                      dnn_list=None, fix_path=None, extra=''):
    pass

    csi_dataloader = CsiDataloader(data_path, train_data_radio=0.9)
    dataset = PathEstNetDataset(csiDataloader=csi_dataloader, dataType=DataType.train, snr_range=snr_range,
                                path_range=path_range, fix_path=fix_path)
    test_dataset = PathEstNetDataset(csiDataloader=csi_dataloader, dataType=DataType.test, snr_range=snr_range,
                                     path_range=path_range, fix_path=fix_path)
    model = DnnPathEst(csiDataloader=csi_dataloader, add_var=add_var, use_true_var=use_true_var, dnn_list=dnn_list,
                       extra=extra)
    criterion = PathEstNetLoss()
    param = TrainParam()
    param.stop_when_test_loss_down_epoch_count = 5
    param.epochs = 20
    param.lr = 0.001
    param.batch_size = 100
    param.log_loss_per_epochs = 1

    train = Train(param, dataset, model, criterion, PathEstNetTee, test_dataset)
    train.train()


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')
    train_pathest_net(data_path='data/imt_2020_32_16_64_400.mat', snr_range=[0, 15], path_range=[5, 15], add_var=True,
                      use_true_var=False, dnn_list=None, fix_path=None, extra='')
    pass
