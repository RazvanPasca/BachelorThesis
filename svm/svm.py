import multiprocessing
import sys
import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from utils.output_utils import compute_confusion_matrix


class SVMTrainConfiguration:
    def __init__(self,
                 max_iterations,
                 c,
                 file_redirect_output,
                 kernel,
                 labels_to_index,
                 window_size,
                 x_train,
                 x_test,
                 y_train,
                 y_test,
                 split_by):
        self.C = c
        self.X_test = x_test
        self.X_train = x_train
        self.file_redirect_output = file_redirect_output
        self.kernel = kernel
        self.labels_to_index = labels_to_index
        self.max_iterations = max_iterations
        self.window_size = window_size
        self.y_test = y_test
        self.y_train = y_train
        self.split_by = split_by


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

    svc = LinearSVC(C=config.C,
                    max_iter=config.max_iterations,
                    dual=False,
                    class_weight='balanced',
                    verbose=0)
    svc.fit(config.X_train, config.y_train)

    conf_mat_test, conf_mat_train = compute_confusion_mats(config, svc)

    if config.file_redirect_output:
        np.array([conf_mat_train, conf_mat_test]) \
            .dump(file_redirect_name + ".npy")
        sys.stdout.close()


def main():
    num_cores = multiprocessing.cpu_count() // 2

    val_perc = 0.2
    max_iterations = 1
    file_redirect_output = True
    window_size = 1000
    cutoff_freq = [10, 100]
    movies_to_keep = [1, 2, 3]
    avsw = False
    split_by_list = ["trials", "random_time_crop", "scramble"]

    data_dict, labels_to_index = load_cat_tf_record(
        CAT_TFRECORDS_PATH_TOBEFORMATED.format(window_size), cutoff_freq)

    datasets = []

    for split_by in split_by_list:
        x_train, x_val, y_train, y_val, new_labels_to_index = new_train_test_split(
            data_dict,
            movies_to_keep,
            val_perc,
            avsw,
            labels_to_index,
            False,
            42,
            split_by,
            window_size)

        datasets.append(
            [np.squeeze(x_train),
             np.squeeze(x_val),
             np.squeeze(y_train),
             np.squeeze(y_val),
             split_by])

    configurations = [SVMTrainConfiguration(
        max_iterations,
        22,
        file_redirect_output,
        "linear",
        labels_to_index,
        window_size,
        X_train,
        X_val,
        Y_train,
        Y_val,
        split_by)
        for X_train, X_val, Y_train, Y_val, split_by in datasets]

    Parallel(n_jobs=num_cores)(delayed(train_svm_with_configuration)(cfg) for cfg in configurations)


if __name__ == '__main__':
    main()
