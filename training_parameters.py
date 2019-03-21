import datetime

import numpy

from datasets.CatLFP import CatLFP


class ModelTrainingParameters:
    def __init__(self, model_path=None, channels_to_keep=[1]):
        self.n_epochs = 5
        self.batch_size = 32
        self.nr_layers = 7
        self.frame_size = 2 ** self.nr_layers
        self.nr_filters = 16
        self.frame_shift = 8
        self.lr = 0.00001
        self.loss = 'CAT'
        self.clip = True
        self.random = True
        self.nr_bins = 256
        self.skip_conn_filters = 32
        self.regularization_coef = 0.0001
        self.normalization = "Zsc"
        self.dataset = CatLFP(channels_to_keep=channels_to_keep, nr_bins=self.nr_bins, normalization=self.normalization)
        self.nr_train_steps = 5  # self.dataset.get_total_length("TRAIN") // self.batch_size
        self.nr_val_steps = 2  # numpy.ceil(0.1 * self.dataset.get_total_length("VAL"))
        self.channels_to_keep = channels_to_keep
        self._get_model_path(model_path)

    def get_model_name(self):
        return "Wavenet_L:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_Norm:{}_{}_Clip:{}_Rnd:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.regularization_coef,
            self.normalization,
            self.loss,
            self.clip,
            self.random)

    def _get_model_path(self, model_path):
        if model_path is None:
            self.model_path = './LFP_models' + "/" + 'Channels:{}'.format(
                str(self.channels_to_keep)) + "/" + self.get_model_name() + '/' + datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M")
        else:
            self.model_path = model_path

    def get_classifying(self):
        return self.loss == "CAT"
