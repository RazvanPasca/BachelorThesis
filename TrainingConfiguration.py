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

        self.set_padding_type()
        self.set_use_vae_flag()

        self._prepare_dataset()
        assert (self.dataset.slice_length < self.frame_size)

        self.nr_output_classes = self.dataset.get_nr_classes()
        self.input_shape = self.dataset.get_input_shape()
        self.condition_on_gamma = self.dataset.condition_on_gamma
        self.slice_length = self.dataset.slice_length
        self.gamma_windows_in_trial = self.dataset_args["gamma_windows_in_trial"]
        self.output_image_size = self.dataset.stimuli_width
        self.stack_channels = self.dataset_args["stack_channels"]

        self.check_model_loss_type()

        self.nr_train_steps = self.dataset.get_training_dataset_size() * self.train_coverage_per_epoch // self.batch_size
        self.nr_val_steps = self.dataset.get_validation_dataset_size() * self.val_coverage_per_epoch // self.batch_size

        self.set_classes_names()
        self._compute_model_path()

    def set_padding_type(self):
        if self.model_type == ModelType.NEXT_TIMESTEP:
            self.padding = 'causal'
        else:
            self.padding = 'same'

    def set_use_vae_flag(self):
        if self.kl_weight is None or self.model_type != ModelType.IMAGE_REC:
            self.use_vae = False
        else:
            self.use_vae = True

    def _prepare_dataset(self):
        self.dataset = self.dataset_class(**self.dataset_args)

    def get_model_name(self):
        return "EncL:{}_Ep:{}_StpEp:{}_Perc:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_ZDim:{}_L2:{}_Loss:{}_GradClip:{}_LPass:{}_DecL:{}_Kl:{}".format(
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
            self.dataset.cutoff_freq,
            self.deconv_layers,
            self.kl_weight)

    def _compute_model_path(self):
        self.model_path = os.path.abspath(os.path.join(
            self.save_path, "{}-{}/Movies:{}/{}-{}-WinL:{}-Stacked:{}/{}/Pid:{}__{}_Seed:{}".format(
                self.model_type,
                "VAE" if self.use_vae else "AE",
                str(self.dataset.conditions_to_keep),
                self.dataset.split_by,
                self.dataset.slicing_strategy,
                self.dataset.slice_length,
                self.stack_channels,
                self.get_model_name(),
                os.getpid(),
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                self.dataset.random_seed)))
        create_dir_if_not_exists(self.model_path)

    def set_classes_names(self):
        self.classes = ["Condition {}".format(i) for i in self.dataset.conditions_to_keep]

    def check_model_loss_type(self):
        if self.model_type == ModelType.CONDITION_CLASSIFICATION or self.model_type == ModelType.SCENE_CLASSIFICATION or self.model_type == ModelType.NEXT_TIMESTEP:
            assert (self.loss == "CE")
        if self.model_type == ModelType.BRIGHTNESS or self.model_type == ModelType.IMAGE_REC or self.model_type == ModelType.EDGES:
            assert (self.loss == "MAE" or self.loss == "MSE")
