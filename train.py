import os
from shutil import copyfile

from callbacks.callback_factory import get_model_callbacks
from models.model_factory import get_model
from train.TrainingConfiguration import TrainingConfiguration
from utils.system_utils import create_dir_if_not_exists
from utils.tf_utils import configure_gpu


def log_training_session(model_params: TrainingConfiguration, model):
    print("Frame size is {}".format(model_params.frame_size))
    print(model_params.get_model_name())
    print('Total training steps:', model_params.nr_train_steps)
    print('Total validation steps:', model_params.nr_val_steps)
    print(model.summary())
    print('Train steps per epoch:', model_params.nr_train_steps)
    print('Val steps per epoch:', model_params.nr_val_steps)


def save_model(model, model_params):
    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


def prepare_logging_folder(model_parameters):
    create_dir_if_not_exists(model_parameters.model_path)
    copyfile(model_parameters.original_config_file_path, os.path.join(model_parameters.model_path, "config_file.py"))


def train_model(model_params: TrainingConfiguration):
    model = get_model(model_params)

    callbacks = get_model_callbacks(model_params)

    log_training_session(model_params, model)
    model.fit_generator(generator=model_params.dataset.train_sample_generator(model_params.batch_size),
                        steps_per_epoch=model_params.nr_train_steps,
                        epochs=model_params.n_epochs,
                        validation_data=model_params.dataset.validation_sample_generator(model_params.batch_size),
                        validation_steps=model_params.nr_val_steps,
                        verbose=1,
                        callbacks=callbacks,
                        use_multiprocessing=False)

    return model


def start_training():
    model_parameters = TrainingConfiguration("training_parameters.py")
    configure_gpu(model_parameters.gpu)
    prepare_logging_folder(model_parameters)
    model = train_model(model_parameters)
    save_model(model, model_parameters)


if __name__ == "__main__":
    start_training()
