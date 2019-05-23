import datetime
import os

from datasets.CatLFPStimuli import CatLFPStimuli
from plot_utils import create_dir_if_not_exists


class ModelTrainingParameters:
    def __init__(self, params_dictionary):
        for k, v in params_dictionary.items():
            setattr(self, k, v)

        self.dataset = CatLFPStimuli()
        self.nr_train_steps = self.dataset.train.size // self.batch_size
        self.nr_val_steps = self.dataset.validation.size // self.batch_size
        self._compute_model_path()
        self.frame_size = 2 ** self.nr_layers

    def get_model_name(self):
        return "EncL:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_Norm:Brute_Loss:MSE_GradClip:{}_LPass:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.regularization_coef,
            self.clip_grad_by_value,
            self.cutoff_freq)

    def _compute_model_path(self):
        self.model_path = os.path.abspath(os.path.join(
            self.save_path, "Labels:{}/{}/Pid:{}__{}_Seed:{}".format(
                str(self.labels_to_keep),
                self.get_model_name(),
                os.getpid(),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                self.random_seed)))
        create_dir_if_not_exists(self.model_path)
