import numpy as np
from keras import callbacks

from model_test_utils.test_ae_model import plot_images_reconstructions
from utils.system_utils import create_dir_if_not_exists


class ReconstructImageCallback(callbacks.Callback):
    def __init__(self, train_batch, val_batch, logging_period, model_path):
        super().__init__()
        self.logging_period = logging_period
        self.epoch_count = 0
        self.train_reconstr_save_path = "{}/reconstructions/train/".format(model_path)
        self.val_reconstr_save_path = "{}/reconstructions/val/".format(model_path)

        create_dir_if_not_exists(self.train_reconstr_save_path)
        create_dir_if_not_exists(self.val_reconstr_save_path)

        self.train_batch, self.val_batch = train_batch, val_batch
        self.nr_train_rec = train_batch[0].shape[0]
        self.full_batch = np.concatenate((self.train_batch[0], self.val_batch[0]), axis=0)

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch_count % self.logging_period == 0:
            reconstructions = self.model.predict(self.full_batch)
            train_reconstr = reconstructions[:self.nr_train_rec]
            val_reconstr = reconstructions[self.nr_train_rec:]

            plot_images_reconstructions(train_reconstr, self.train_batch[1],
                                        self.train_reconstr_save_path, self.epoch_count)
            plot_images_reconstructions(val_reconstr, self.val_batch[1],
                                        self.val_reconstr_save_path, self.epoch_count)
        self.epoch_count += 1
