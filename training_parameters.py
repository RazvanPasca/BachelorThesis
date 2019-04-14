import os
import datetime
import json
import numpy as np

LOCAL_CONFIG_PATH = "train_params_cfg.json"


class ModelTrainingParameters:
    def __init__(self, model_path=None):
        self.save_path = None
        self.clip_grad_by_value = None
        self.loss = None
        self.regularization_coef = None
        self.skip_conn_filters = None
        self.nr_filters = None
        self.lr = None
        self.n_epochs = None
        self.nr_layers = 0
        self.batch_size = None
        self.normalization = None
        self.nr_bins = None
        self.channels_to_keep = None
        self.conditions_to_keep = None
        self.movies_to_keep = None
        self.n_epochs = None
        self.random_seed = 42

        if model_path is not None:
            config_path = os.path.join(model_path, "train_params_cfg.json")
            if os.path.exists(config_path):
                self._load_configuration_from_json(config_path)
            else:
                self._load_configuration_from_json(LOCAL_CONFIG_PATH)
        else:
            self._load_configuration_from_json(LOCAL_CONFIG_PATH)

        self.frame_size = 2 ** self.nr_layers

        klass = getattr(getattr(__import__("datasets"), self.dataset), self.dataset)
        self._prepare_dataset(klass)

        self.nr_train_steps = np.ceil(
            self.train_coverage_per_epoch * self.dataset.get_total_length("TRAIN")) // self.batch_size
        self.nr_val_steps = np.ceil(
            self.val_coverage_per_epoch * self.dataset.get_total_length("VAL")) // self.batch_size

        self._compute_model_path(model_path)

    def _prepare_dataset(self, klass):
        if "CatLFP" == self.dataset:
            self.dataset = klass(movies_to_keep=self.movies_to_keep,
                                 channels_to_keep=self.channels_to_keep,
                                 nr_bins=self.nr_bins,
                                 normalization=self.normalization,
                                 cutoff_freq=self.cutoff_freq,
                                 random_seed=self.random_seed)
        else:
            self.dataset = klass(conditions_to_keep=self.conditions_to_keep,
                                 channels_to_keep=self.channels_to_keep,
                                 nr_bins=self.nr_bins,
                                 normalization=self.normalization,
                                 cutoff_freq=self.cutoff_freq,
                                 random_seed=self.random_seed)

    def _load_configuration_from_json(self, config_path):
        with open(config_path, 'r') as f:
            json_config = json.loads(f.read())
        for prop, val in json_config.items():
            setattr(self, prop, val)

    def get_model_name(self):
        if self.get_classifying() == 2:
            loss = self.loss + ":{}_RegW:{}_SfmaxW:{}".format(self.nr_bins,
                                                                        self.multiloss_weights["Regression"],
                                                                        self.multiloss_weights["Sfmax"])
        elif self.get_classifying() == 1:
            loss = self.loss + ":{}".format(self.nr_bins)
        else:
            loss = self.loss
        return "WvNet_L:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_Norm:{}_Loss:{}_GradClip:{}_LPass:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.regularization_coef,
            self.normalization,
            loss,
            self.clip_grad_by_value,
            self.cutoff_freq,
        )

    def _compute_model_path(self, model_path):
        if model_path is None:
            self.model_path = os.path.abspath(os.path.join(self.save_path,
                                                           "{}/Movies:{}/Channels:{}/{}/Pid:{}_{}_Seed:{}".format(
                                                               type(self.dataset).__name__,
                                                               str(self.movies_to_keep),
                                                               str(self.channels_to_keep),
                                                               self.get_model_name(),
                                                               os.getpid(),
                                                               datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                                               self.random_seed)
                                                           )
                                              )
        else:
            self.model_path = model_path

    def get_classifying(self):

        if self.loss == "MSE_CE" or self.loss == "MAE_CE":
            return 2
        elif self.loss == "CE":
            return 1
        else:
            return -1

    def serialize_to_json(self, path):
        attrs_dict = dict(self.__dict__)
        attrs_dict["dataset"] = type(attrs_dict["dataset"]).__name__
        with open(os.path.join(path, "train_params_cfg.json"), 'w+') as g:
            g.write(json.dumps(attrs_dict, sort_keys=True, indent=4))
