model_parameters = {
    "nr_layers": 6,
    "skip_conn_filters": 128,
    "nr_filters": 256,
    "lr": 1e-05,
    "batch_size": 16,
    "loss": "MSE",
    "n_epochs": 100,
    "clip_grad_by_value": 5,
    "regularization_coef": 0.001,
    "random_seed": 42,
    "logging_period": 3,
    "cutoff_freq": [
        0,
        80
    ],
    "nr_rec": 12,
    "train_coverage_per_epoch": 0.001,
    "val_coverage_per_epoch": 0.001,
    "movies_to_keep": None,
    "labels_to_keep": None,
    "gpu": 0,
    "save_path": "/data2/razpa/SceneGen",
}
