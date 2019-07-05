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
        "stack_channels": True,
        "use_mu_law": False,
        "number_of_bins": 255,
        "val_percentage": 0.2,
        "channels_to_keep": None,
        "conditions_to_keep": [
            0,1,2
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
    "batch_size": 64,
    "nr_layers": 7,
    "nr_filters": 128,
    "skip_conn_filters": 256,
    "lr": 1e-05,
    "loss": "MAE",
    "clip_value": 5,
    "regularization_coef": 0.00,
    "logging_period": 3,
    "train_coverage_per_epoch": 0.1,
    "val_coverage_per_epoch": 0.5,
    "save_path": "/data2/razpa/Results_after_refactor/new_vae",
    "nr_rec": 18,
    "generative_samples": 36,
    "kl_weight": 0.001,
    "deconv_layers": [512, 256, 128, 64, 32],
    "z_dim": 100,
    "gpu": 1
}
