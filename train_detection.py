from train import *


def train_detection_net_3(data_path: str, snr_range: list, layer=32, svm='v', train_layer_step=None, modulation='bpsk',
                          use_layer_total_loss=True, save=True, reload=True,
                          retrain=False, extra=''):
    refinements = [.5, .1, .05, .01, .001]
    csi_dataloader = CsiDataloader(data_path, factor=1)
    model = DetectionNetModel(csi_dataloader, layer, svm=svm, modulation=modulation, extra=extra)
    test_dataset = DetectionNetDataset(csi_dataloader, DataType.test, snr_range, modulation)

    criterion = DetectionNetLoss(use_layer_total_mse=use_layer_total_loss)
    param = TrainParam()
    param.batch_size = 100
    param.use_scheduler = False
    epochs = 10

    dataset = DetectionNetDataset(csi_dataloader, DataType.train, snr_range, modulation)
    train = Train(param, dataset, model, criterion, DetectionNetTee, test_dataset)
    if train_layer_step is None:
        train_layer_range = [layer]
    else:
        train_layer_range = list(range(1, layer + 1, train_layer_step))
        if layer not in train_layer_range:
            train_layer_range.append(layer)
    if not retrain and reload and os.path.exists(train.get_save_path()):
        model_infos = torch.load(train.get_save_path())
        if 'train_state' in model_infos:
            current_train_layer = model_infos['train_state']['train_layer']
            logging.warning('load train state:{}'.format(model_infos['train_state']))
            train_layer_range = train_layer_range[train_layer_range.index(current_train_layer):]

    for layer_num in train_layer_range:
        logging.info('Fine tune layer:{}'.format(layer_num))
        train.param.epochs = epochs

        learn_rate = train.param.lr
        for factor in refinements:
            train.param.lr = learn_rate * factor
            model.set_training_layer(layer_num, False)
            train.train(save=save, reload=reload,
                        ext_log='snr:{},model:{},lr:{}'.format(-1, model.get_train_state_str(), param.lr))
            train.reset_current_epoch()
        train.param.lr = learn_rate


def train_detection_net_2(data_path: str, snr_range: list, layer, modulation='bpsk', svm='v', save=True, reload=True,
                          retrain=False):
    refinements = [.5, .1, .01]
    csi_dataloader = CsiDataloader(data_path, factor=1)
    model = DetectionNetModel(csi_dataloader, layer, svm=svm, modulation=modulation)
    test_dataset = DetectionNetDataset(csi_dataloader, DataType.test, snr_range, modulation)

    criterion = DetectionNetLoss()
    param = TrainParam()
    param.batch_size = 100
    param.use_scheduler = False

    dataset = DetectionNetDataset(csi_dataloader, DataType.train, snr_range, modulation)
    train = Train(param, dataset, model, criterion, DetectionNetTee, test_dataset)
    current_train_layer = 1
    over_fix_forward = False
    epochs = 10
    if not retrain and reload and os.path.exists(train.get_save_path()):
        model_infos = torch.load(train.get_save_path())
        if 'train_state' in model_infos:
            current_train_layer = model_infos['train_state']['train_layer']
            over_fix_forward = not model_infos['train_state']['fix_forward']
            logging.warning('load train state:{}'.format(model_infos['train_state']))

    for layer_num in range(current_train_layer, model.layer_nums + 1):
        if not over_fix_forward:
            logging.info('training layer:{}'.format(layer_num))
            train.param.epochs = epochs
            train.param.lr = 0.001
            model.set_training_layer(layer_num, True)
            train.train(save=save, reload=reload,
                        ext_log='snr:{},model:{}'.format(-1, model.get_train_state_str()))
            train.reset_current_epoch()

        over_fix_forward = False
        logging.info('Fine tune layer:{}'.format(layer_num))
        train.param.epochs = epochs

        learn_rate = train.param.lr
        for factor in refinements:
            train.param.lr = learn_rate * factor
            model.set_training_layer(layer_num, False)
            train.train(save=save, reload=reload,
                        ext_log='snr:{},model:{},lr:{}'.format(-1, model.get_train_state_str(), param.lr))
            train.reset_current_epoch()


def train_detection_net(data_path: str, training_snr: list, modulation='qpsk', svm='v', save=True, reload=True,
                        retrain=False):
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

    csi_dataloader = CsiDataloader(data_path, factor=1)
    model = DetectionNetModel(csi_dataloader, csi_dataloader.n_t, svm=svm, modulation=modulation)
    if retrain and os.path.exists(Train.get_save_path_from_model(model)):
        model_info = torch.load(Train.get_save_path_from_model(model))
        model_info['snr'] = training_snr[0]
        model_info['epoch'] = 0
        model.set_training_layer(1, True)
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
                train.param.epochs = 100
                train.param.lr = 0.001
                model.set_training_layer(layer_num, True)
                train.train(save=save, reload=reload,
                            ext_log='snr:{},model:{}'.format(snr, model.get_train_state_str()))
                train.reset_current_epoch()

            over_fix_forward = False
            logging.info('Fine tune layer:{}'.format(layer_num))
            train.param.epochs = 100

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
    train_detection_net_3('data/spatial_mu_ULA_64_32_64_100_l10_11.mat', [0, 10], layer=32, svm='v',
                          train_layer_step=None, use_layer_total_loss=True, modulation='qpsk', retrain=True)
    # train_detection_net_2('data/spatial_mu_ULA_64_32_64_100_l10_11.mat', [5, 20], layer=32, svm='v', modulation='bpsk',
    #                       retrain=True)
    # train_detection_net_3('data/spatial_mu_ULA_64_32_64_400_l10_11.mat', [5, 20], modulation='qpsk', layer=32, svm='v',
    #                       retrain=True)
