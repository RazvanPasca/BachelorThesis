import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from datasets.LFPDataset import LFPDataset
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED


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
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))


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


def load_cat_tf_record(path):
    cat_scenes_dataset = load_tf_record(path)
    labels = set([sample[0] for sample in cat_scenes_dataset])
    labels_to_index = {}
    for index, label in enumerate(labels):
        labels_to_index[label] = index
    x = []
    y = []
    for sample in cat_scenes_dataset:
        x.append(sample[4])
        y.append(labels_to_index[sample[0]])
    x = np.array(x)
    x /= x.max()
    return x, y, labels_to_index


if __name__ == '__main__':
    # x, y = load_cat_dataset(CAT_DATASET_PATH, label='movie_frame')
    # x, y = load_mouse_dataset(MOUSEACH_DATASET_PATH, label='contrast', ignore_channels=True)

    file_redirect = "output_windowsize_{}.txt"
    max_iterations = 100000

    for window_size in [1000, 800, 400, 200]:
        if not (file_redirect is None):
            sys.stdout = open(file_redirect.format(window_size), 'w+')

        x, y, labels_to_index = load_cat_tf_record(CAT_TFRECORDS_PATH_TOBEFORMATED.format(window_size))
        print(labels_to_index)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

        for kernel in ["linear", "poly", "rbf"]:
            for i in np.arange(-1, 2, 0.33):
                print("Window size={}".format(window_size))
                print("C={}".format(10 ** i))
                print("Kernel={}".format(kernel))
                print("Max iterations={}".format(max_iterations))
                svc = SVC(kernel=kernel,
                          C=10 ** i,
                          max_iter=max_iterations,
                          cache_size=15000,
                          class_weight='balanced',
                          verbose=False)

                svc.fit(X_train, y_train)

                print("TRAIN CONFUSION MATRIX:")
                compute_confusion_matrix(svc, X_train, y_train)

                if not len(y_test) is 0:
                    print("TEST CONFUSION MATRIX:")
                    compute_confusion_matrix(svc, X_test, y_test)
