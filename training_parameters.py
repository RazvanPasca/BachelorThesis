from datasets import CatDataset
from datasets.datasets_utils import SplitStrategy, SlicingStrategy, ModelType

training_parameters = {
    "dataset_class": CatDataset,
    "dataset_args": {
        "random_seed": 42,
        "slice_length": 100,
        "model_type": ModelType.IMAGE_REC,
        "slicing_strategy": SlicingStrategy.CONSECUTIVE,
        "split_by": SplitStrategy.TRIALS,
        "stack_channels": False,
        "use_mu_law": False,
        "number_of_bins": 255,
        "val_percentage": 0.2,
        "channels_to_keep": [0, 1, 2, 3, ],
        "conditions_to_keep": [
            0, 1, 2
        ],
        # "orientations_to_keep": [0, 1],
        # "contrasts_to_keep": [0, 1],
        "trials_to_keep": None,
        "cutoff_freq": None,
        "condition_on_gamma": False,
        "gamma_windows_in_trial": [
            [
                400,
                700
            ],
            [
                1000,
                1350
            ],
            [
                1700,
                2040
            ],
            [
                2350,
                2700
            ]
        ]
    },
    "n_epochs": 400,
    "batch_size": 32,
    "nr_layers": 6,
    "nr_filters": 16,
    "skip_conn_filters": 16,
    "lr": 1e-05,
    "loss": "MAE",
    "clip_value": 5,
    "regularization_coef": 0.001,
    "logging_period": 3,
    "train_coverage_per_epoch": 0.01,
    "val_coverage_per_epoch": 0.5,
    "save_path": "./Refactored",
    "nr_rec": 18,
    "generative_samples": 25,
    "kl_weight": None,
    "deconv_layers": [512, 256, 128, 64, 32],
    "z_dim": 10,
    "gpu": 0
}
