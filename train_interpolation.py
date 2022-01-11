from loader import InterpolationNetDataset
from model import CBDNetSFModel, InterpolationNetLoss, InterpolationNetTee
from train import *
from utils import InterpolationMethodTransformChuck, Transform
from utils.InterpolationMethod import get_transformChuckMethod_fix_path


def loss_nmse_func(train: Train):
    nmse = None
    for items in train.test_dataloader:
        tee = train.teeClass(items)
        tee.set_model_output(train.model.eval()(*tee.get_model_input()))
        h_hat, _ = train.model.eval()(*tee.get_model_input())
        h = items[1]
        nmse_ = (torch.abs(h - h_hat) ** 2).sum(-1).sum(-1) / (torch.abs(h) ** 2).sum(-1).sum(-1)
        if nmse is None:
            nmse = nmse_
        else:
            nmse = torch.cat((nmse, nmse_), dim=0)
        if nmse is not None:
            nmse = 10 * torch.log10(nmse.mean())
            return nmse.item()
        else:
            return None

    return 0


def train_interpolation_net(csi_dataloader:CsiDataloader, snr_range: list, chuckMethod: InterpolationMethodTransformChuck, add_var,
                            n_f=0, conv=6, channel=64, kernel_size=(3, 3), extra=''):
    dataset = InterpolationNetDataset(csi_dataloader, DataType.train, snr_range, chuckMethod)
    test_dataset = InterpolationNetDataset(csi_dataloader, DataType.test, snr_range, chuckMethod)

    loss_nmse_func_ = None
    # if save_loss_nmse_data:
    #     loss_nmse_func_ = loss_nmse_func

    model = CBDNetSFModel(csiDataloader=csi_dataloader, chuck_name=chuckMethod.get_key_name(), add_var=add_var, n_f=n_f,
                          conv=conv, channel=channel, kernel_size=kernel_size, extra=extra)
    criterion = InterpolationNetLoss()
    param = TrainParam()
    param.stop_when_test_loss_down_epoch_count = 5
    param.epochs = 50
    param.lr = 0.001
    param.batch_size = 100
    param.log_loss_per_epochs = 1

    train = Train(param, dataset, model, criterion, InterpolationNetTee, test_dataset)
    train.train(loss_data_func=loss_nmse_func_)


if __name__ == '__main__':
    logging.basicConfig(level=20, format='%(asctime)s-%(levelname)s-%(message)s')

    # spatial mu ula
    # block
    csi_dataloader = CsiDataloader('data/spatial_mu_ULA_64_32_64_100_l10_11.mat', train_data_radio=0.9)
    cp = 16
    chuckMethod10_nf0 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=10, n_f=0, cp=cp)
    chuckMethod10_nf1 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=10, n_f=1, cp=cp)
    chuckMethod10_nf2 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=10, n_f=2, cp=cp)
    chuckMethod10_nf3 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=10, n_f=3, cp=cp)
    chuckMethod_cp_nf0 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=cp, n_f=0, cp=cp)
    chuckMethod_None_nf0 = get_transformChuckMethod_fix_path(csi_dataloader, Transform.dft, fix_path=csi_dataloader.n_sc, n_f=0, cp=cp)

    # base
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod10_nf0, add_var=True, n_f=0, conv=6, channel=64, extra='')
    # without var
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod10_nf0, add_var=False, n_f=0, conv=6, channel=64, extra='')
    # without dft chuck
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod_None_nf0, add_var=True, n_f=0, conv=6, channel=64, extra='')
    # cp chuck
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod_cp_nf0, add_var=True, n_f=0, conv=6, channel=64, extra='')

    # comb nf=1
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod10_nf1, add_var=True, n_f=1, conv=6, channel=64, extra='')

    # comb nf=2
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod10_nf2, add_var=True, n_f=2, conv=6, channel=64, extra='')
    # comb nf=3
    train_interpolation_net(csi_dataloader=csi_dataloader, snr_range=[5, 25],
                            chuckMethod=chuckMethod10_nf3, add_var=True, n_f=3, conv=6, channel=64, extra='')
