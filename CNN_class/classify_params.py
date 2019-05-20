import datetime

model_params = {
    "nr_classes": -1,
    "n_epochs": 400,
    "batch_size": 64,
    "nr_layers": 6,
    "nr_filters": 16,
    "skip_conn_filters": 16,
    "lr": 1e-05,
    "loss": "CE",
    "clip_grad_by_value": 5,
    "regularization_coef": 0.0,
    "val_coverage_per_epoch": 0.3,
    "cutoff_freq": [0.01, 80],
    "gpu": 0,
    "shuffle_seed": 40,
    "logging_period": 3,
}


def get_model_name(model_params):
    return "./CatClassification/WvNet_L:{}_Ep:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_" \
           "Norm:ZscBrute_Loss:CE_GradClip:{}_LPass:{}_Seed:{}_TrainTest:{}/{}".format(
        model_params["nr_layers"],
        model_params["n_epochs"],
        model_params["lr"],
        model_params["batch_size"],
        model_params["nr_filters"],
        model_params["skip_conn_filters"],
        model_params["regularization_coef"],
        model_params["clip_grad_by_value"],
        model_params["cutoff_freq"],
        model_params["shuffle_seed"],
        model_params["val_coverage_per_epoch"],
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
