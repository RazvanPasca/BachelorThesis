import numpy as np
import tensorflow as tf

from signal_analysis.signal_utils import get_filter_type, filter_input_sequence
from utils.tf_utils import shuffle_indices


def new_train_test_split(data_dict, movies_to_keep, val_perc, AvsW, labels_to_index, concatenate_channels, seed,
                         split_by, win_size):
    np.random.seed(seed)

    dataset = [list(data_dict[movie].values()) for movie in movies_to_keep]  # list of 3 values
    new_labels_to_index = labels_to_index.copy()

    X_train, X_val, Y_train, Y_val, new_labels_to_index = filter_by_labels(AvsW, dataset, concatenate_channels,
                                                                           new_labels_to_index, split_by, val_perc,
                                                                           win_size)

    if len(X_train.shape) == 2:
        X_train, X_val = np.expand_dims(X_train, axis=2), np.expand_dims(X_val, axis=2)

    return X_train, X_val, Y_train, Y_val, new_labels_to_index


def filter_by_labels(AvsW, dataset, concat_channels, new_labels_to_index, split_by, split_perc, win_size):
    nr_trials = 20
    nr_channels = 47

    if split_by.lower() == "trials":
        train_trials, val_trials = shuffle_indices(nr_trials, split_perc, False)

        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        for movie in dataset:
            for trial_index in train_trials:
                for example in movie[trial_index]:
                    X_train, Y_train = append_by_concat_channels(X_train, Y_train, concat_channels, example, win_size)

        for movie in dataset:
            for trial_index in val_trials:
                for example in movie[trial_index]:
                    X_val, Y_val = append_by_concat_channels(X_val, Y_val, concat_channels, example, win_size)

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

    elif split_by.lower() == "channels":
        train_channels, val_channels = shuffle_indices(nr_channels, split_perc, True)

        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        for movie in dataset:
            for trial in movie:
                for example in trial:
                    nr_examples_indiv_channels = example[0].size // win_size
                    for i in range(nr_examples_indiv_channels):
                        if i in train_channels:
                            X_train.append(example[0][i * win_size: (i + 1) * win_size])
                            Y_train.append(example[1])
                        if i in val_channels:
                            X_val.append(example[0][i * win_size: (i + 1) * win_size])
                            Y_val.append(example[1])

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)


    elif split_by.lower() == "scramble":
        X = []
        Y = []

        for movie in dataset:
            for trial in movie:
                for example in trial:
                    X, Y = append_by_concat_channels(X, Y, concat_channels, example, win_size)

        X, Y = np.stack(X), np.stack(Y)
        train_indices, val_indices = shuffle_indices(X.shape[0], split_perc, False)
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_val, Y_val = X[val_indices], Y[val_indices]


    elif split_by.lower() == "random_time_crop":
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []

        for movie in dataset:
            for trial in movie:
                train_examples, val_examples = shuffle_indices(len(trial), 0.2, get_sets=True)
                examples_count = 0
                for example in trial:
                    if examples_count in train_examples:
                        X_train, Y_train = append_by_concat_channels(X_train, Y_train, concat_channels, example,
                                                                     win_size)
                    else:
                        X_val, Y_val = append_by_concat_channels(X_val, Y_val, concat_channels, example, win_size)
                    examples_count += 1

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)


    elif split_by.lower() == "time_crop":
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []

        for movie in dataset:
            train_examples, val_examples = shuffle_indices(len(movie[0]), 0.2, get_sets=True)
            for trial in movie:
                examples_count = 0
                for example in trial:
                    if examples_count in train_examples:
                        X_train, Y_train = append_by_concat_channels(X_train, Y_train, concat_channels, example,
                                                                     win_size)
                    else:
                        X_val, Y_val = append_by_concat_channels(X_val, Y_val, concat_channels, example, win_size)
                    examples_count += 1

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

    # TODO implement label filtering on the new obtained tensors
    # Currently just the all setting works

    return X_train, X_val, Y_train, Y_val, new_labels_to_index


def append_by_concat_channels(X, Y, concat_channels, example, win_size):
    if concat_channels:
        # pdb.set_trace()
        X.append(example[0].reshape((47, -1)).transpose())
        Y.append([example[1]])
    else:
        nr_examples_indiv_channels = example[0].size // win_size
        for i in range(nr_examples_indiv_channels):
            X.append(example[0][i * win_size: (i + 1) * win_size])
            Y.append(example[1])

    return X, Y


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

            signal = np.array(example.features.feature['signal'].float_list.value)

            reconstructed_data.append((scene, trial, movie, signal))
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
            trial_dict_for_movie[sample[1]].append((filter_input_sequence(sample[3], cuttof_freq, filter_type), label))
        else:
            trial_dict_for_movie[sample[1]] = [(filter_input_sequence(sample[3], cuttof_freq, filter_type), label)]

    for movie_key, trial_dict in data_dict.copy().items():
        reindexed_trial_dict = {}
        for i, old_trial_key in enumerate(trial_dict.keys()):
            reindexed_trial_dict[i] = trial_dict[old_trial_key]
        data_dict[movie_key] = reindexed_trial_dict
    return data_dict, labels_to_index
