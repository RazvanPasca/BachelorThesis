model_parameters = {
    "nr_layers": 6,
    "skip_conn_filters": 32,
    "nr_filters": 32,
    "lr": 1e-06,
    "batch_size": 64,
    "loss": "MAE",
    "model_output_type": "classify_movies",  # brightness / edges / dcgan / classify_movies
    "n_epochs": 300,
    "clip_grad_by_value": 5,
    "regularization_coef": 0.003,
    "z_dim": 100,
    "train_val_split": 0.2,
    "random_seed": 42,
    "logging_period": 3,
    "cutoff_freq": None,
    "nr_rec": 18,
    "split_by": "trials",  # one of slices or trials
    "slice_length": 40,  # if slicing_strategy == fixed, this gives the length of slice
    "slicing_strategy": "consecutive",  # consecutive or random, should use consecutive for now
    "train_coverage_per_epoch": 0.1,
    "val_coverage_per_epoch": 0.5,
    "movies_to_keep": [0, 1, 2],  # one of [0,1,2]
    "labels_to_keep": None,
    "gpu": 0,
    "save_path": "./SceneGen",
}
