from datasets import CatDataset
from datasets.datasets_utils import SplitStrategy, SlicingStrategy, ModelType

training_parameters = {
    "dataset_class": CatDataset,
    "dataset_args": {
        "random_seed": 42,
        "slice_length": 100,
        "slicing_strategy": SlicingStrategy.CONSECUTIVE,
        "stack_channels": False,
        "use_mu_law": False,
        "model_type": ModelType.BRIGHTNESS,
        "number_of_bins": 255,
        "val_percentage": 0.2,
        "split_by": SplitStrategy.TRIALS,
        "channels_to_keep": None,
        "movies_to_keep": [
            0, 1, 2
        ],
        "trials_to_keep": None,
        "cutoff_freq": [
            1, 80
        ],
        # "condition_on_gamma": False,
        # "gamma_windows_in_trial": [
        #     [
        #         400,
        #         700
        #     ],
        #     [
        #         1000,
        #         1350
        #     ],
        #     [
        #         1700,
        #         2040
        #     ],
        #     [
        #         2350,
        #         2700
        #     ]
        # ]
    },
    "n_epochs": 400,
    "batch_size": 32,
    "nr_layers": 6,
    "nr_filters": 16,
    "skip_conn_filters": 16,
    "lr": 1e-05,
    "loss": "CE",
    "clip_value": 5,
    "regularization_coef": 0.001,
    "logging_period": 3,
    "train_coverage_per_epoch": 0.005,
    "val_coverage_per_epoch": 0.25,
    "save_path": "./LFP_models",
    "nr_rec": 18,
    "kl_weight": None,
    "deconv_layers": [],
    "z_dim": 10,
    "gpu": 0
}
