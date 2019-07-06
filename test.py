import os
import sys

from keras.engine.saving import load_model
from matplotlib import pyplot as plt

from models.model_factory import get_model
from train.TrainingConfiguration import TrainingConfiguration


def load_model_from_folder(model_folder):
    config_path = os.path.join(model_folder, "config_file.py")
    model_parameters = TrainingConfiguration(config_path)
    model = get_model(model_parameters)
    model.load_weights(os.path.join(model_folder, "best_model.h5"))
    return model, model_parameters


def test_model(model_folder):
    model, params = load_model_from_folder(model_folder)
    gen = params.dataset.validation_sample_generator(128, return_address=False)
    X_val, Y_val = next(gen)
    Y_pred = model.predict(X_val)
    plt.plot(Y_pred, Y_val)
    plt.show()


if __name__ == "__main__":
    test_model(sys.argv[1])
