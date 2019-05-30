import datetime
import os

model_params = {
    "n_epochs": 400,
    "batch_size": 64,
    "nr_layers": 6,
    "nr_filters": 16,
    "skip_conn_filters": 16,
    "lr": 1e-05,
    "loss": "CE",
    "clip_grad_by_value": 5,
    "regularization_coef": 0.001,
    "val_perc": 0.2,
    "cutoff_freq": None,
    "movies_to_keep": [1, 2, 3],  # list with [1,2,3] or w/e
    "gpu": 0,
    "concatenate_channels": False,
    "shuffle_seed": 42,
    "logging_period": 3,
    "window_size": 1000,
    "ClassW": True,
    "split_by": "time_crop",  # one of trials, scramble, time_crop, random_time_crop
    "AvsW": "all",  # one of 1v1, merge or all # time_crop doesnt work properly???
    "save_path": "."
}


def get_model_name(model_params):
    return "{}/CatClassification/Movies:{}/SplitBy:{}/AvsW:{}/WvNet_L:{}_Concat:{}_Ep:{}_Lr:{}_BS:{}_Fltrs:{}_SkipFltrs:{}_L2:{}_" \
           "Norm:ZscBrute_GradClip:{}_LPass:{}_Seed:{}_TrainTest:{}_WinSize:{}_ClassW:{}/PID:{}__Date:{}".format(
        model_params["save_path"],
        model_params["movies_to_keep"],
        model_params["split_by"],
        model_params["AvsW"],
        model_params["nr_layers"],
        model_params["concatenate_channels"],
        model_params["n_epochs"],
        model_params["lr"],
        model_params["batch_size"],
        model_params["nr_filters"],
        model_params["skip_conn_filters"],
        model_params["regularization_coef"],
        model_params["clip_grad_by_value"],
        model_params["cutoff_freq"],
        model_params["shuffle_seed"],
        model_params["val_perc"],
        model_params["window_size"],
        model_params["ClassW"],
        os.getpid(),
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
