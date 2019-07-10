from datasets import CatDataset
from datasets.datasets_utils import SplitStrategy, SlicingStrategy, ModelType

training_parameters = {
    "dataset_class": CatDataset,  # CatDataset/MouseControlDataset/MouseAChDataset
    "dataset_args": {
        "random_seed": 42,
        "slice_length": 100,
        "model_type": ModelType.BRIGHTNESS,
        "relative_difference": True,
        # BRIGHTNESS/IMAGE_REC/EDGES/CONDITION_CLASSIFICATION/SCENE_CLASSIFICATION
        "slicing_strategy": SlicingStrategy.CONSECUTIVE,  # CONSECUTIVE or RANDOM
        "split_by": SplitStrategy.TRIALS,  # TRIALS or SLICES
        "stack_channels": True,
        "use_mu_law": False,
        "number_of_bins": 255,
        "val_percentage": 0.2,
        "channels_to_keep": None,
        "conditions_to_keep": [
            0, 1, 2
        ],
        # "orientations_to_keep": None,
        # "contrasts_to_keep": None,  # 0 is 25%, 1 is 100%, 2 is 50%
        "blur_images": False,
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
    "nr_filters": 16,
    "skip_conn_filters": 16,
    "lr": 1e-05,
    "use_dil": False,
    "loss": "MAE",
    "clip_value": 5,
    "regularization_coef": 0.001,
    "logging_period": 3,
    "train_coverage_per_epoch": 0.003,
    "val_coverage_per_epoch": 0.005,
    "save_path": "./Results_after_refactor/",
    "nr_rec": 18,
    "generative_samples": 49,
    "kl_weight": 0.001,
    "deconv_layers": [128, 64, 32, 16, 8],
    "z_dim": 10,
    "gpu": 0
}
