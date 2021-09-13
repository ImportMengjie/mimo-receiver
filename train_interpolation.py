from train import *


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


def train_interpolation_net(data_path: str, snr_range: list, pilot_count, noise_level_conv=4, noise_channel=32,
                            noise_dnn=(2000, 200, 50), denoising_conv=6, denoising_channel=64, kernel_size=(3, 3),
                            use_two_dim=True, use_true_sigma=False, only_train_noise_level=False, fix_noise=False,
                            extra='', dft_chuck=0, use_dft_padding=False, save_loss_nmse_data=False):
    assert not (only_train_noise_level and use_true_sigma)
    assert not (only_train_noise_level and fix_noise)
    csi_dataloader = CsiDataloader(data_path, train_data_radio=0.9)
    dataset = InterpolationNetDataset(csi_dataloader, DataType.train, snr_range, pilot_count)
    test_dataset = InterpolationNetDataset(csi_dataloader, DataType.test, snr_range, pilot_count)

    loss_nmse_func_ = None
    if save_loss_nmse_data:
        loss_nmse_func_ = loss_nmse_func

    model = CBDNetSFModel(csi_dataloader, pilot_count, noise_level_conv=noise_level_conv, noise_channel=noise_channel,
                          noise_dnn=noise_dnn, denoising_conv=denoising_conv, denoising_channel=denoising_channel,
                          kernel_size=kernel_size, use_two_dim=use_two_dim, use_true_sigma=use_true_sigma,
                          only_return_noise_level=only_train_noise_level, extra=extra, dft_chuck=dft_chuck,
                          use_dft_padding=use_dft_padding)
    criterion = InterpolationNetLoss(only_train_noise_level=only_train_noise_level, use2dim=use_two_dim)
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

    # block
    train_interpolation_net(data_path='data/spatial_mu_ULA_64_32_64_400_l10_11.mat', snr_range=[5, 25], pilot_count=63,
                            noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
                            denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                            only_train_noise_level=False, fix_noise=True, extra='', dft_chuck=10,
                            use_dft_padding=False, save_loss_nmse_data=True)
    # comb nf=1
    train_interpolation_net(data_path='data/spatial_mu_ULA_64_32_64_400_l10_11.mat', snr_range=[5, 25], pilot_count=31,
                            noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
                            denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                            only_train_noise_level=False, fix_noise=True, extra='', dft_chuck=0,
                            use_dft_padding=True)

    # comb nf=2
    train_interpolation_net(data_path='data/spatial_mu_ULA_64_32_64_400_l10_11.mat', snr_range=[5, 25], pilot_count=20,
                            noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
                            denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                            only_train_noise_level=False, fix_noise=True, extra='', dft_chuck=0,
                            use_dft_padding=True)
    # comb nf=3
    train_interpolation_net(data_path='data/spatial_mu_ULA_64_32_64_400_l10_11.mat', snr_range=[5, 25], pilot_count=15,
                            noise_level_conv=4, noise_channel=32, noise_dnn=(2000, 200, 50), denoising_conv=6,
                            denoising_channel=64, kernel_size=(3, 3), use_two_dim=True, use_true_sigma=True,
                            only_train_noise_level=False, fix_noise=True, extra='', dft_chuck=0,
                            use_dft_padding=True)
