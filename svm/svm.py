import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVR

from datasets import CatDataset
from datasets.datasets_utils import SequenceAddress
from utils.output_utils import compute_confusion_matrix


class SVMTrainConfiguration:
    def __init__(self,
                 max_iterations,
                 c,
                 file_redirect_output,
                 kernel,
                 labels_to_index,
                 split_by,
                 window_size,
                 x_train,
                 x_test,
                 y_train,
                 y_test):
        self.split_by = split_by
        self.window_size = window_size
        self.C = c
        self.X_test = x_test
        self.X_train = x_train
        self.file_redirect_output = file_redirect_output
        self.kernel = kernel
        self.labels_to_index = labels_to_index
        self.max_iterations = max_iterations
        self.y_test = y_test
        self.y_train = y_train


def compute_confusion_matrix(y_pred, y):
    conf_mat = confusion_matrix(y, y_pred)
    classif_report = classification_report(y, y_pred)
    print(conf_mat)
    print(classif_report)
    return conf_mat, classif_report


def print_configuration(config):
    print(config.labels_to_index)
    print("Window size={}".format(config.window_size))
    print("C={}".format(config.C))
    print("Kernel={}".format(config.kernel))
    print("Max iterations={}".format(config.max_iterations))


def compute_confusion_mats(config, svc):
    print("TRAIN CONFUSION MATRIX:")
    conf_mat_train, _ = compute_confusion_matrix(svc.predict(config.X_train), config.y_train)
    print("TEST CONFUSION MATRIX:")
    conf_mat_test, _ = compute_confusion_matrix(svc.predict(config.X_test), config.y_test)
    return conf_mat_test, conf_mat_train


def train_svm_with_configuration(config):
    file_redirect_name = "SplitBy-{}-WS-{}-C-{}-K-{}-MaxIt-{}".format(
        config.split_by,
        config.window_size,
        config.C,
        config.kernel,
        config.max_iterations)

    if config.file_redirect_output:
        sys.stdout = open(file_redirect_name, 'w+')

    print_configuration(config)

    svc = LinearSVR(C=config.C,
                    max_iter=config.max_iterations,
                    dual=False,
                    verbose=0)
    svc.fit(config.X_train, config.y_train)

    conf_mat_test, conf_mat_train = compute_confusion_mats(config, svc)

    if config.file_redirect_output:
        np.array([conf_mat_train, conf_mat_test]) \
            .dump(file_redirect_name + ".npy")
        sys.stdout.close()


def main(dataset, max_iter, nr_cores, training_parameters):
    file_redirect_output = True

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    train_set = dataset.prepared_data["TRAIN"]
    test_set = dataset.prepared_data["VAL"]

    for i, condition in enumerate(train_set):
        for j, trial in enumerate(condition):
            for k, channel in enumerate(trial):
                for l, slice in enumerate(channel):
                    seq_addr = SequenceAddress(i, trial, channel,
                                               dataset.slice_indexes["TRAIN"][l] * dataset.slice_length,
                                               dataset.slice_length,
                                               "TRAIN")
                    X_train.append(slice)
                    Y_train.append(dataset._get_y_value_for_sequence(seq_addr))

    for i, condition in enumerate(test_set):
        for j, trial in enumerate(condition):
            for k, channel in enumerate(trial):
                for l, slice in enumerate(channel):
                    seq_addr = SequenceAddress(i, trial, channel,
                                               dataset.slice_indexes["VAL"][l] * dataset.slice_length,
                                               dataset.slice_length,
                                               "VAL")
                    X_test.append(slice)
                    Y_test.append(dataset._get_y_value_for_sequence(seq_addr))

    configurations = [SVMTrainConfiguration(max_iter,
                                            C,
                                            file_redirect_output,
                                            "linear",
                                            None,
                                            training_parameters["dataset_args"]["split_by"],
                                            training_parameters["dataset_args"]["slice_length"],
                                            X_train,
                                            X_test,
                                            Y_train,
                                            Y_test) for C in [0, 3, 10, 30, 100, 1000]]

    Parallel(n_jobs=nr_cores)(delayed(train_svm_with_configuration)(cfg) for cfg in configurations)


if __name__ == '__main__':
    from training_parameters import training_parameters

    nr_cores = 1

    dataset = CatDataset(**training_parameters["dataset_args"])
    max_iter = 100
    main(dataset, max_iter, nr_cores, training_parameters)
