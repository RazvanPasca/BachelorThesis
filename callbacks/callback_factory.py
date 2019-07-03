from keras.callbacks import CSVLogger, ModelCheckpoint

import TrainingConfiguration
from callbacks.ConfusionMatrixPlotter import ConfusionMatrixPlotter
from callbacks.GeneratedSequencePlotCallback import GeneratedSequencePlotCallback
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from callbacks.ReconstructImageCallback import ReconstructImageCallback
from callbacks.VaeCallback import VaeCallback
from datasets.datasets_utils import ModelType


def get_model_callbacks(model_args: TrainingConfiguration):
    callbacks = get_common_callbacks(model_args)

    if model_args.model_type == ModelType.CONDITION_CLASSIFICATION:
        train_batch = model_args.dataset.get_train_plot_examples(model_args.nr_train_steps)
        val_batch = model_args.dataset.get_train_plot_examples(model_args.nr_val_steps)
        conf_matrix_callback = ConfusionMatrixPlotter(train_batch,
                                                      val_batch,
                                                      model_args.classes,
                                                      model_args.model_path,
                                                      model_args.logging_period,
                                                      normalize=True)
        callbacks.append(conf_matrix_callback)

    if model_args.model_type == ModelType.NEXT_TIMESTEP:
        plot_generated_signals_callback = GeneratedSequencePlotCallback(model_args,
                                                                        model_args.logging_period,
                                                                        nr_predictions=-1,
                                                                        starting_point=0,
                                                                        all_reset_indices=[
                                                                            model_args.dataset.gamma_windows_in_trial,
                                                                            [model_args.dataset.trial_length - 1]],
                                                                        nr_plot_rows=3)
        callbacks.append(plot_generated_signals_callback)

    if model_args.model_type == ModelType.IMAGE_REC:
        train_batch = model_args.dataset.get_train_plot_examples(model_args.nr_rec)
        val_batch = model_args.dataset.get_train_plot_examples(model_args.nr_rec)

        reconstruct_image_callback = ReconstructImageCallback(train_batch,
                                                              val_batch,
                                                              model_args.logging_period,
                                                              model_args.model_path)
        callbacks.append(reconstruct_image_callback)

    if model_args.use_vae:
        vae_sampling_callback = VaeCallback(model_args)
        callbacks.append(vae_sampling_callback)

    return callbacks


def get_common_callbacks(model_args):
    callbacks = []

    log_callback = CSVLogger(model_args.model_path + "/session_log.csv")
    callbacks.append(log_callback)

    if model_args.model_type in [ModelType.CONDITION_CLASSIFICATION, ModelType.SCENE_CLASSIFICATION]:
        metrics = ["loss", "acc"]
    elif model_args.use_vae:
        metrics = ["loss", "reconstruction_loss"]
    else:
        metrics = ["loss"]

    metric_callback = MetricsPlotCallback(model_args.model_path, metrics)
    callbacks.append(metric_callback)

    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_args.model_path),
                                          monitor="val_loss", save_best_only=True)
    callbacks.append(save_model_callback)

    return callbacks
