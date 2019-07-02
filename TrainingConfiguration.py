import datetime
import os

from datasets.datasets_utils import ModelType
from utils.plot_utils import create_dir_if_not_exists


class TrainingConfiguration:
    def __init__(self, params_dictionary):
        self.dataset_args = None
        self.save_path = None
        self.cutoff_freq = None
        self.clip_grad_by_value = None
        self.regularization_coef = None
        self.val_coverage_per_epoch = None
        self.train_coverage_per_epoch = None
        self.n_epochs = None
        self.batch_size = None
        self.gpu = None
        self.z_dim = None
        self.model_type = None
        self.nr_layers = 0
        self.nr_filters = None
        self.skip_conn_filters = None
        self.reg_coeff = None
        self.lr = None
        self.clip_value = None
        self.loss = None
        self.deconv_layers = None
        self.kl_weight = None

        for k, v in params_dictionary.items():
            setattr(self, k, v)

        self.frame_size = 2 ** self.nr_layers
        self.model_type = self.dataset_args["model_type"]

        if self.model_type == ModelType.NEXT_TIMESTEP:
            self.padding = 'causal'
        else:
            self.padding = 'same'

        if self.kl_weight is None:
            self.use_vae = False
        else:
            self.use_vae = True

        self._prepare_dataset()
        assert (self.dataset.slice_length > self.frame_size)

        self.nr_output_classes = self.dataset.get_nr_classes()
        self.input_shape = self.dataset.get_input_shape()

        self.nr_train_steps = (
                self.dataset.get_training_dataset_size() // self.batch_size * self.train_coverage_per_epoch)
        self.nr_val_steps = (
                self.dataset.get_validation_dataset_size() // self.batch_size * self.val_coverage_per_epoch)

        self._compute_model_path()

    def _prepare_dataset(self):
        self.dataset = self.dataset_class(**self.dataset_args)

    def get_model_name(self):
        return "EncL:{}_Ep:{}_StpEp:{}_Perc:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_ZDim:{}_L2:{}_Norm:Brute_Loss:{}_GradClip:{}_LPass:{}".format(
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
            self.save_path, "{}/Movies:{}/SplitBy:{}-Strategy:{}-WinL:{}-/{}/Pid:{}__{}_Seed:{}".format(
                self.model_type,
                str(self.dataset.movies_to_keep),
                self.dataset.split_by,
                self.dataset.slicing_strategy,
                self.dataset.slice_length,
                self.get_model_name(),
                os.getpid(),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                self.dataset.random_seed)))
        create_dir_if_not_exists(self.model_path)
