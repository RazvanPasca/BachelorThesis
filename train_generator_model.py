from keras.callbacks import CSVLogger, ModelCheckpoint

from WavenetGenerator.test_model import test_model
from WavenetGenerator.ModelTrainingParameters import ModelTrainingParameters
from WavenetGenerator.wavenet_generator_model import get_wavenet_model
from callbacks.GenErrorPlotCallback import GenErrorPlotCallback as PlotCallback
from callbacks.TboardCallbackWrapper import TboardCallbackWrapper as TensorBoardWrapper
from utils.plot_utils import create_dir_if_not_exists
from utils.tf_utils import configure_gpu


def log_training_session(model_params):
    print("Frame size is {}".format(model_params.frame_size))
    print(model_params.get_model_name())
    print('Total training steps:', model_params.nr_train_steps)
    print('Total validation steps:', model_params.nr_val_steps)


def train_model(model_params):
    log_training_session(model_params)

    model = get_wavenet_model(nr_filters=model_params.nr_filters,
                              input_shape=(model_params.frame_size, 2),
                              nr_layers=model_params.nr_layers,
                              lr=model_params.lr,
                              loss=model_params.loss,
                              clipvalue=model_params.clip_grad_by_value,
                              skip_conn_filters=model_params.skip_conn_filters,
                              regularization_coef=model_params.regularization_coef,
                              nr_output_classes=model_params.nr_bins,
                              multiloss_weights=model_params.multiloss_weights)

    # TODO make activation callback work with multiple output

    tensorboard_callback = TensorBoardWrapper(
        batch_gen=model_params.dataset.validation_sample_generator(model_params.frame_size,
                                                                   model_params.batch_size,
                                                                   model_params.get_classifying()),
        nb_steps=10,
        log_dir=model_params.model_path,
        write_graph=True,
        histogram_freq=5,
        batch_size=model_params.batch_size)

    # tensorboard_callback = TensorBoard(log_dir=model_params.model_path, write_graph=True, )
    log_callback = CSVLogger(model_params.model_path + "/session_log.csv")
    plot_generated_signals_callback = PlotCallback(model_params,
                                                   model_params.logging_period,
                                                   nr_predictions=-1,
                                                   starting_point=0,
                                                   all_reset_indices=[model_params.dataset.gamma_windows_in_trial,
                                                                      [model_params.dataset.trial_length - 1]])

    path_models_ = model_params.model_path + "/models/"
    create_dir_if_not_exists(path_models_)
    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_params.model_path),
                                          monitor="val_loss",
                                          save_best_only=True)

    model.fit_generator(model_params.dataset.train_sample_generator(model_params.frame_size,
                                                                    model_params.batch_size,
                                                                    model_params.get_classifying()),
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=model_params.dataset.validation_sample_generator(model_params.frame_size,
                                                                                         model_params.batch_size,
                                                                                         model_params.get_classifying()),
                        validation_steps=model_params.nr_val_steps,
                        verbose=2,
                        callbacks=[tensorboard_callback, plot_generated_signals_callback, log_callback,
                                   save_model_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = ModelTrainingParameters()
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
    test_model(model_parameters)
