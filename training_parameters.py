import os
import datetime
import json
import numpy

LOCAL_CONFIG_PATH = "train_params_cfg.json"


class ModelTrainingParameters:
    def __init__(self, model_path=None):
        self.save_path = None
        self.random = None
        self.clip = None
        self.loss = None
        self.regularization_coef = None
        self.skip_conn_filters = None
        self.nr_filters = None
        self.lr = None
        self.n_epochs = None
        self.nr_layers = None
        self.batch_size = None
        self.normalization = None
        self.nr_bins = None
        self.channels_to_keep = None
        self.n_epochs = None
        self.frame_shift = None

        if model_path is not None:
            config_path = os.path.join(model_path, "train_params_cfg.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    json_config = json.loads(f.read())
                for prop, val in json_config.items():
                    setattr(self, prop, val)
        else:
            with open(LOCAL_CONFIG_PATH, 'r') as f:
                json_config = json.loads(f.read())
            for prop, val in json_config.items():
                setattr(self, prop, val)

        klass = getattr(getattr(__import__("datasets"), self.dataset), self.dataset)

        self.frame_size = 2 ** self.nr_layers
        self.dataset = klass(self.channels_to_keep, self.nr_bins, self.normalization)
        self.nr_train_steps = numpy.ceil(0.1 * self.dataset.get_total_length("TRAIN")) // self.batch_size
        self.nr_val_steps = numpy.ceil(0.1 * self.dataset.get_total_length("VAL")) // self.batch_size
        self._compute_model_path(model_path)

    def get_model_name(self):
        return "WvNet_L:{}_Ep:{}_StpEp:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_Norm:{}_{}_Clip:{}_Rnd:{}".format(
            self.nr_layers,
            self.n_epochs,
            self.nr_train_steps,
            self.lr,
            self.batch_size,
            self.nr_filters,
            self.skip_conn_filters,
            self.regularization_coef,
            self.normalization,
            self.loss + ":{}".format(self.nr_bins) if self.get_classifying() else self.loss,
            self.clip,
            self.random)

    def _compute_model_path(self, model_path):
        if model_path is None:
            self.model_path = os.path.abspath(os.path.join(self.save_path,
                                                           "{}/Channels:{}/{}/{}".format(type(self.dataset).__name__,
                                                                                         str(self.channels_to_keep),
                                                                                         self.get_model_name(),
                                                                                         datetime.datetime.now().strftime(
                                                                                             "%Y-%m-%d %H:%M"))))
        else:
            self.model_path = model_path

    def get_classifying(self):
        return self.loss == "CAT"

    def serialize_to_json(self, path):
        attrs_dict = self.__dict__
        attrs_dict["dataset"] = type(attrs_dict["dataset"]).__name__
        with open(os.path.join(path, "train_params_cfg.json"), 'w+') as g:
            g.write(json.dumps(attrs_dict, sort_keys=True, indent=4))
