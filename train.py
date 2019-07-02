from training_parameters import training_parameters
from TrainingConfiguration import TrainingConfiguration
from models.model_factory import get_model
from utils.plot_utils import create_dir_if_not_exists
from utils.tf_utils import configure_gpu


def log_training_session(model_params: TrainingConfiguration):
    print("Frame size is {}".format(model_params.frame_size))
    print(model_params.get_model_name())
    print('Total training steps:', model_params.nr_train_steps)
    print('Total validation steps:', model_params.nr_val_steps)


def train_model(model_params: TrainingConfiguration):
    log_training_session(model_params)

    path_models_ = model_params.model_path + "/models/"
    create_dir_if_not_exists(path_models_)

    model = get_model(model_params)

    callbacks = None  # get_callbacks_for_model(model_params)

    model.fit_generator(generator=model_params.dataset.train_sample_generator(model_params.batch_size),
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=model_params.dataset.validation_sample_generator(model_params.batch_size),
                        validation_steps=model_params.nr_val_steps,
                        verbose=2,
                        callbacks=callbacks)

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = TrainingConfiguration(training_parameters)
    configure_gpu(model_parameters.gpu)
    train_model(model_parameters)
