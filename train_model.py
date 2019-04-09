from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint

from plot_utils import PlotCallback, TensorBoardWrapper
from test_model import test_model
from tf_utils import configure_gpu
from training_parameters import ModelTrainingParameters
from model import get_basic_generative_model


def log_training_session(model_params):
    print("Frame size is {}".format(model_params.frame_size))
    print(model_params.get_model_name())
    print('Total training steps:', model_params.nr_train_steps)
    print('Total validation steps:', model_params.nr_val_steps)


def train_model(model_params):
    log_training_session(model_params)

    model = get_basic_generative_model(model_params.nr_filters,
                                       model_params.frame_size,
                                       model_params.nr_layers,
                                       lr=model_params.lr,
                                       loss=model_params.loss,
                                       clipvalue=model_params.clip_grad_by_value,
                                       skip_conn_filters=model_params.skip_conn_filters,
                                       regularization_coef=model_params.regularization_coef,
                                       nr_output_classes=model_params.nr_bins)

    tensor_board_callback = TensorBoardWrapper(
        batch_gen=model_params.dataset.validation_frame_generator(model_params.frame_size,
                                                                  model_params.batch_size,
                                                                  model_params.get_classifying()),
        nb_steps=1,
        log_dir=model_params.model_path,
        write_graph=True,
        histogram_freq=1,
        batch_size=model_params.batch_size)
    log_callback = CSVLogger(model_params.model_path + "/session_log.csv")
    plot_figure_callback = PlotCallback(model_params, 5, nr_predictions_steps=100,
                                        starting_point=1200 - model_params.frame_size)
    save_model_callback = ModelCheckpoint(
        filepath="{}/best_model.h5".format(model_params.model_path), monitor="val_loss",
        save_best_only=True)

    model.fit_generator(
        model_params.dataset.train_frame_generator(model_params.frame_size,
                                                   model_params.batch_size,
                                                   model_params.get_classifying()),
        steps_per_epoch=model_params.nr_train_steps, epochs=model_params.n_epochs,
        validation_data=model_params.dataset.validation_frame_generator(model_params.frame_size,
                                                                        model_params.batch_size,
                                                                        model_params.get_classifying()),
        validation_steps=model_params.nr_val_steps,
        verbose=2,
        callbacks=[tensor_board_callback, plot_figure_callback, log_callback, save_model_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = ModelTrainingParameters()
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
    test_model(model_parameters)
