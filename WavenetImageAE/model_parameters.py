model_parameters = {
    "nr_layers": 6,
    "skip_conn_filters": 16,
    "nr_filters": 16,
    "lr": 1e-05,
    "batch_size": 32,
    "loss": "MSE",
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
    "train_coverage_per_epoch": 0.001,
    "val_coverage_per_epoch": 0.001,
    "movies_to_keep": [0, 1, 2],  # one of [0,1,2]
    "labels_to_keep": None,
    "gpu": 0,
    "save_path": "./SceneGen",
}
