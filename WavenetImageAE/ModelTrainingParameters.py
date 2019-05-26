import datetime
import os

from datasets.CatLFPStimuli import CatLFPStimuli
from utils.plot_utils import create_dir_if_not_exists


class ModelTrainingParameters:
    def __init__(self, params_dictionary):
        for k, v in params_dictionary.items():
            setattr(self, k, v)

        self.dataset = CatLFPStimuli(val_perc=self.train_val_split,
                                     cutoff_freq=self.cutoff_freq,
                                     movies_to_keep=self.movies_to_keep)
        self.nr_train_steps = (self.dataset.train.size // self.batch_size * self.train_coverage_per_epoch) // \
                              self.dataset.number_of_channels
        self.nr_val_steps = (self.dataset.validation.size // self.batch_size * self.val_coverage_per_epoch) // \
                            self.dataset.number_of_channels
        self._compute_model_path()
        self.frame_size = 2 ** self.nr_layers

    def get_model_name(self):
        return "EncL:{}_Ep:{}_StpEp:{}_Perc:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{} ZDim:{}_L2:{}_Norm:Brute_Loss:{}_GradClip:{}_LPass:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.train_coverage_per_epoch,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.z_dim,
            self.regularization_coef,
            self.loss,
            self.clip_grad_by_value,
            self.cutoff_freq)

    def _compute_model_path(self):
        self.model_path = os.path.abspath(os.path.join(
            self.save_path, "Movies:{}/{}/Pid:{}__{}_Seed:{}".format(
                str(self.movies_to_keep),
                self.get_model_name(),
                os.getpid(),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                self.random_seed)))
        create_dir_if_not_exists(self.model_path)
