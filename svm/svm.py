import multiprocessing
import pdb
import sys

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC

from datasets.loaders import new_train_test_split, load_cat_tf_record
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED


def compute_confusion_matrix(svc, x, y):
    y_pred = svc.predict(x)
    conf_mat = confusion_matrix(y, y_pred)
    clas_rep = classification_report(y, y_pred)
    print(conf_mat)
    print(clas_rep)
    return conf_mat, clas_rep


def main():
    max_iterations = 100000
    file_redirect_output = True

    num_cores = multiprocessing.cpu_count() // 2

    window_size = 1000
    cutoff_freq = [10, 100]
    movies_to_keep = [1, 2, 3]
    val_perc = 0.2
    AvsW = False
    split_by_list = ["trials", "random_time_crop", "scramble"]

    data_dict, labels_to_index = load_cat_tf_record(
        CAT_TFRECORDS_PATH_TOBEFORMATED.format(window_size), cutoff_freq)

    X_train_list = []
    X_val_list = []
    Y_train_list = []
    Y_val_list = []

    for split_by in split_by_list:
        X_train, X_val, Y_train, Y_val, new_labels_to_index = new_train_test_split(data_dict, movies_to_keep, val_perc,
                                                                                   AvsW,
                                                                                   labels_to_index,
                                                                                   False,
                                                                                   42,
                                                                                   split_by,
                                                                                   window_size)

        X_train_list.append(np.squeeze(X_train))
        X_val_list.append(np.squeeze(X_val))
        Y_train_list.append(np.squeeze(Y_train))
        Y_val_list.append(np.squeeze(Y_val))
    pdb.set_trace()
    results = Parallel(n_jobs=num_cores)(delayed(train_svm_with_params)
                                         (22, X_val, X_train, file_redirect_output, "linear", labels_to_index,
                                          max_iterations, window_size, Y_val, Y_train, split_by)
                                         for X_train, X_val, Y_train, Y_val, split_by
                                         in
                                         zip(X_train_list, X_val_list, Y_train_list, Y_val_list, split_by_list))


def train_svm_with_params(C,
                          X_test,
                          X_train,
                          file_redirect_output,
                          kernel,
                          labels_to_index,
                          max_iterations,
                          window_size,
                          y_test,
                          y_train,
                          split_by):
    if file_redirect_output:
        file_redirect_name = "SplitBy-{}-WS-{}-C-{}-K-{}-MaxIt-{}".format(split_by, window_size, C, kernel,
                                                                          max_iterations)
        sys.stdout = open(file_redirect_name, 'w+')

    print(labels_to_index)
    print("Window size={}".format(window_size))
    print("C={}".format(C))
    print("Kernel={}".format(kernel))
    print("Max iterations={}".format(max_iterations))
    svc = LinearSVC(C=C,
                    max_iter=max_iterations,
                    dual=False,
                    class_weight='balanced',
                    verbose=0)
    svc.fit(X_train, y_train)
    print("TRAIN CONFUSION MATRIX:")
    conf_mat_train, _ = compute_confusion_matrix(svc, X_train, y_train)
    print("TEST CONFUSION MATRIX:")
    conf_mat_test, _ = compute_confusion_matrix(svc, X_test, y_test)
    data = np.array([conf_mat_train, conf_mat_test])
    if file_redirect_output:
        data.dump(file_redirect_name + ".npy")
    if file_redirect_output:
        sys.stdout.close()


if __name__ == '__main__':
    # x, y = load_cat_dataset(CAT_DATASET_PATH, label='movie_frame')
    # x, y = load_mouse_dataset(MOUSEACH_DATASET_PATH, label='contrast', ignore_channels=True)

    main()
