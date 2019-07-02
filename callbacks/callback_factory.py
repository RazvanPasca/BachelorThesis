from keras.callbacks import CSVLogger, ModelCheckpoint

import TrainingConfiguration
from callbacks.GeneratedSequencePlotCallback import GeneratedSequencePlotCallback
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from callbacks.ReconstructImageCallback import ReconstructImageCallback
from datasets.datasets_utils import ModelType


def get_model_callbacks(model_args: TrainingConfiguration):
    callbacks = []

    log_callback = CSVLogger(model_args.model_path + "/session_log.csv")
    callbacks.append(log_callback)

    metrics = ["loss", "acc"] if model_args.model_type in [ModelType.CONDITION_CLASSIFICATION,
                                                           ModelType.SCENE_CLASSIFICATION] else ["loss"]
    metric_callback = MetricsPlotCallback(model_args.model_path, metrics)
    callbacks.append(metric_callback)

    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_args.model_path),
                                          monitor="val_loss", save_best_only=True)
    callbacks.append(save_model_callback)

    if model_args.model_type == ModelType.NEXT_TIMESTEP:
        plot_generated_signals_callback = GeneratedSequencePlotCallback(model_args,
                                                                        model_args.logging_period,
                                                                        nr_predictions=-1,
                                                                        starting_point=0,
                                                                        all_reset_indices=[
                                                                            # model_args.dataset.gamma_windows_in_trial,
                                                                            [model_args.dataset.trial_length - 1]],
                                                                        nr_plot_rows=3)
        callbacks.append(plot_generated_signals_callback)

    if model_args.model_type == ModelType.IMAGE_REC:
        train_images_to_reconstr = next(model_args.dataset.train_sample_generator(model_args.nr_rec))
        val_images_to_reconstr = next(model_args.dataset.val_sample_generator(model_args.nr_rec))

        reconstruct_image_callback = ReconstructImageCallback(train_images_to_reconstr,
                                                              val_images_to_reconstr,
                                                              model_args.logging_period,
                                                              model_args.model_path)
        callbacks.append(reconstruct_image_callback)

    return callbacks
