import os

import numpy as np
from keras.engine.saving import load_model
from matplotlib import pyplot as plt

from train.TrainingConfiguration import TrainingConfiguration

MY_MODEL_PATH = "Pid:3428__2019-07-07 03:09_Seed:42"


def load_model_from_folder(model_folder):
    config_path = os.path.join(model_folder, "config_file.py")
    model_parameters = TrainingConfiguration(config_path)
    model = load_model(os.path.join(model_folder, "best_model.h5"))
    return model, model_parameters


def test_model(nr_samples, model_folder):
    model, params = load_model_from_folder(model_folder)
    labels = ["Condition 0", "Condition 1", "Condition 2"]

    gen = params.dataset.validation_sample_generator(nr_samples, return_address=True)
    X_val, Y_val, addresses = next(gen)
    Y_pred = model.predict(X_val)
    abs_dif = np.abs(Y_pred - Y_val)

    for condition in range(3):
        Y_val_list = []
        Y_pred_list = []
        for i, address in enumerate(addresses):
            if address.condition == condition:
                Y_val_list.append(Y_val[i])
                Y_pred_list.append(Y_pred[i])

        plt.scatter(Y_val_list, Y_pred_list, label=labels[condition])

    plt.legend()
    fontsize = 15
    plt.title("Test prediction MAE:{:.4}, prediction error std:{:.4}".format(np.mean(abs_dif), np.std(abs_dif)),
              fontsize=fontsize)
    plt.xlabel("Actual values", fontsize=15)
    plt.ylabel("Predicted values", fontsize=15)
    plt.show()

    gen = params.dataset.train_sample_generator(nr_samples, return_address=True)
    X_val, Y_val, addresses = next(gen)
    Y_pred = model.predict(X_val)
    abs_dif = np.abs(Y_pred - Y_val)

    for condition in range(3):
        Y_val_list = []
        Y_pred_list = []
        for i, address in enumerate(addresses):
            if address.condition == condition:
                Y_val_list.append(Y_val[i])
                Y_pred_list.append(Y_pred[i])

        plt.scatter(Y_val_list, Y_pred_list, label=labels[condition])

    plt.legend()
    plt.title("Train prediction MAE:{:.4}, prediction error std:{:.4}".format(np.mean(abs_dif), np.std(abs_dif)),
              fontsize=fontsize)
    plt.xlabel("Actual values", fontsize=15)
    plt.ylabel("Predicted values", fontsize=15)
    plt.show()


if __name__ == "__main__":
    nr_samples = 3000
    test_model(nr_samples, MY_MODEL_PATH)
