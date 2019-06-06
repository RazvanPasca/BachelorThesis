model_parameters = {
    "nr_layers": 6,
    "skip_conn_filters": 16,
    "nr_filters": 16,
    "lr": 3e-05,
    "batch_size": 32,
    "loss": "MAE",
    "model": "regression",  # regression for brightness regression or dcgan for reconstructions
    "n_epochs": 100,
    "clip_grad_by_value": 5,
    "regularization_coef": 0.001,
    "z_dim": 100,
    "train_val_split": 0.2,
    "random_seed": 42,
    "logging_period": 3,
    "cutoff_freq": None,
    "nr_rec": 18,
    "split_by": "slices",  # one of slices or trials
    "slice_length": 100,  # if slicing_strategy == fixed, this gives the length of slice
    "slicing_strategy": "fixed",  # fixed or random
    "train_coverage_per_epoch": 0.01,
    "val_coverage_per_epoch": 0.01,
    "movies_to_keep": [0, 1, 2],  # one of [0,1,2]
    "labels_to_keep": None,
    "gpu": 0,
    "save_path": "./SceneGen",
}
