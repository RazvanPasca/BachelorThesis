from keras.callbacks import ModelCheckpoint, TensorBoard

from datasets.CatLFPStimuli import CatLFPStimuli
from plot_utils import create_dir_if_not_exists
from tf_utils import configure_gpu
from wavenet_dcgan import get_wavenet_dcgan_model

import numpy as np


class TempModelTrainingParameters:
    def __init__(self):
        self.model_path = ".\\test_wavenet_conv"
        self.save_path = ".\\test_wavenet_conv"
        self.clip_grad_by_value = 5
        self.regularization_coef = 0.001
        self.skip_conn_filters = 128
        self.nr_filters = 128
        self.lr = 1e-05
        self.nr_layers = 4
        self.batch_size = 8
        self.n_epochs = 100
        self.random_seed = 42
        self.frame_size = 2 ** self.nr_layers
        self.dataset = CatLFPStimuli()
        self.gpu = 0
        self.logging_period = 3
        self.nr_train_steps = 100
        self.nr_val_steps = 10
        self.normalization = None
        self.zs = None
        self.channels_to_keep = None
        self.conditions_to_keep = None
        self.multiloss_weights = None
        self.loss = None


class TensorReconBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        x, y = next(self.batch_gen)
        self.validation_data = (x, y.reshape(-1, 1), np.ones(self.batch_size))
        return super().on_epoch_end(epoch, logs)


def train_model(model_params):
    model = get_wavenet_dcgan_model(nr_filters=model_params.nr_filters,
                                    input_shape=(model_params.frame_size, 47),
                                    nr_layers=model_params.nr_layers,
                                    lr=model_params.lr,
                                    loss=model_params.loss,
                                    clipvalue=model_params.clip_grad_by_value,
                                    skip_conn_filters=model_params.skip_conn_filters,
                                    regularization_coef=model_params.regularization_coef,
                                    nr_output_classes=model_params.zs,
                                    multiloss_weights=model_params.multiloss_weights)

    print(model.summary())
    # TODO make activation callback work with multiple output

    tensorboard_callback = TensorReconBoardWrapper(
        batch_gen=model_params.dataset.validation_frame_generator(model_params.frame_size,
                                                                  model_params.batch_size),
        nb_steps=10,
        log_dir=model_params.model_path,
        write_graph=True,
        histogram_freq=5,
        batch_size=model_params.batch_size)

    # tensorboard_callback = TensorBoard(log_dir=model_params.model_path, write_graph=True, )
    # log_callback = CSVLogger(model_params.model_path + "/session_log.csv")
    # plot_figure_callback = PlotCallback(model_params,
    #                                     model_params.logging_period,
    #                                     nr_predictions=-1,
    #                                     starting_point=0,
    #                                     all_reset_indices=[model_params.dataset.gamma_windows_in_trial,
    #                                                        [model_params.dataset.trial_length - 1]])

    path_models_ = model_params.model_path + "/models/"
    create_dir_if_not_exists(path_models_)
    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_params.model_path),
                                          monitor="val_loss",
                                          save_best_only=True)

    model.fit_generator(model_params.dataset.train_frame_generator(model_params.frame_size,
                                                                   model_params.batch_size),
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=model_params.dataset.validation_frame_generator(model_params.frame_size,
                                                                                        model_params.batch_size),
                        validation_steps=model_params.nr_val_steps,
                        verbose=2,
                        callbacks=[tensorboard_callback, save_model_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = TempModelTrainingParameters()
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
    # test_model(model_parameters)
