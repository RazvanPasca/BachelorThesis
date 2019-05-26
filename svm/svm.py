import multiprocessing
import sys

import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datasets.LFPDataset import LFPDataset
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from signal_analysis.signal_utils import get_filter_type, filter_input_sample


def load_cat_dataset(path, label='condition'):
    dataset = LFPDataset(path, normalization='Zsc')

    x = []
    y = []
    for condition_id in range(0, dataset.number_of_conditions):
        for i in range(0, dataset.channels.shape[1] // dataset.trial_length):
            if dataset.stimulus_conditions[i] == condition_id + 1:
                trial = dataset.channels[:,
                        (i * dataset.trial_length):((i * dataset.trial_length) + dataset.trial_length)]
                if label == 'condition':
                    x.append(trial.reshape(-1))
                    y.append(condition_id)
                elif label == 'channel':
                    for i, c in enumerate(trial):
                        x.append(c)
                        y.append(i)
                else:
                    return None
    return x, y


def load_mouse_dataset(path, label='orientation', ignore_channels=False):
    dataset = LFPDataset(path, normalization='Zsc')
    x = []
    y = []
    for stimulus_condition in dataset.stimulus_conditions:
        index = int(stimulus_condition['Trial']) - 1
        events = [{'timestamp': dataset.event_timestamps[4 * index + i],
                   'code': dataset.event_codes[4 * index + i]} for i in range(4)]
        trial = dataset.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)][:-1]
        # removed heart rate

        if label == 'orientation':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1) // 3)
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1) // 3)
        elif label == 'contrast':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1) % 3)
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1) % 3)
        elif label == 'condition':
            if ignore_channels:
                for i, c in enumerate(trial):
                    x.append(c)
                    y.append((int(stimulus_condition['Condition number']) - 1))
            else:
                x.append(trial.reshape(-1))
                y.append((int(stimulus_condition['Condition number']) - 1))
        elif label == 'channel':
            for i, c in enumerate(trial):
                x.append(c)
                y.append(i)
        else:
            return None
    return x, y


def compute_confusion_matrix(svc, x, y):
    y_pred = svc.predict(x)
    conf_mat = confusion_matrix(y, y_pred)
    clas_rep = classification_report(y, y_pred)
    print(conf_mat)
    print(clas_rep)
    return conf_mat, clas_rep


def load_tf_record(path):
    reconstructed_data = []

    record_iterator = tf.python_io.tf_record_iterator(path=path)

    for string_record in record_iterator:

        example = tf.train.Example()
        try:
            example.ParseFromString(string_record)

            scene = example.features.feature['scene'].bytes_list.value[0]

            trial = int(example.features.feature['trial']
                        .int64_list
                        .value[0])

            movie = int(example.features.feature['movie']
                        .int64_list
                        .value[0])

            channel = int(example.features.feature['channel']
                          .int64_list
                          .value[0])

            signal = (example.features.feature['signal'].float_list.value)

            reconstructed_data.append((scene, trial, movie, channel, signal))
        except:
            print("An exception occurred in sample {}".format(string_record))
    return reconstructed_data


def load_cat_tf_record(path, cuttof_freq=None):
    data_dict = {1: {},
                 2: {},
                 3: {}}
    cat_scenes_dataset = load_tf_record(path)

    labels = set([sample[0] for sample in cat_scenes_dataset])
    labels_to_index = {}
    for index, label in enumerate(labels):
        labels_to_index[label] = index

    filter_type = get_filter_type(cuttof_freq)
    for sample in cat_scenes_dataset:
        trial_dict_for_movie = data_dict[sample[2]]
        label = labels_to_index[sample[0]]
        if sample[1] in trial_dict_for_movie:
            trial_dict_for_movie[sample[1]].append((filter_input_sample(sample[4], cuttof_freq, filter_type), label))
        else:
            trial_dict_for_movie[sample[1]] = [(filter_input_sample(sample[4], cuttof_freq, filter_type), label)]

    for movie_key, trial_dict in data_dict.copy().items():
        reindexed_trial_dict = {}
        for i, old_trial_key in enumerate(trial_dict.keys()):
            reindexed_trial_dict[i] = trial_dict[old_trial_key]
        data_dict[movie_key] = reindexed_trial_dict
    return data_dict, labels_to_index


def main():
    max_iterations = 1
    file_redirect_output = True
    # what are your inputs, and what operation do you want to
    # perform on each input. For example...
    gammas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    Cs = [22, 24, 26, 16, 18, 20]

    cart_prod = [(gamma, c) for c in Cs for gamma in gammas]

    print(cart_prod)

    num_cores = multiprocessing.cpu_count()

    x, y, labels_to_index = load_cat_tf_record(CAT_TFRECORDS_PATH_TOBEFORMATED.format(1000))
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

    results = Parallel(n_jobs=num_cores)(delayed(train_svm_with_params)(
        gamma, C, X_test, X_train, file_redirect_output, "linear", labels_to_index, max_iterations,
        1000, y_test, y_train
    ) for gamma, C in cart_prod)


def train_svm_with_params(gamma,
                          X_test,
                          X_train,
                          file_redirect_output,
                          kernel,
                          labels_to_index,
                          max_iterations,
                          window_size,
                          y_test,
                          y_train):
    C = 22
    if file_redirect_output:
        file_redirect_name = "WS-{}-C-{}-gamma-{}-K-{}-MaxIt-{}".format(window_size, C, gamma, kernel, max_iterations)
        sys.stdout = open(file_redirect_name, 'w+')
    print(labels_to_index)
    print("Window size={}".format(window_size))
    print("C={}".format(C))
    print("gamma={}".format(gamma))
    print("Kernel={}".format(kernel))
    print("Max iterations={}".format(max_iterations))
    svc = SVC(kernel=kernel,
              C=C,
              gamma=gamma,
              max_iter=max_iterations,
              cache_size=15000,
              class_weight='balanced',
              verbose=False)
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
