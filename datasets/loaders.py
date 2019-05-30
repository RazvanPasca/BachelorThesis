import numpy as np

from utils.tf_utils import shuffle_indices


def new_train_test_split(data_dict, movies_to_keep, val_perc, AvsW, labels_to_index, concatenate_channels, seed,
                         split_by):
    np.random.seed(seed)

    dataset = [list(data_dict[movie].values()) for movie in movies_to_keep]  # list of 3 values
    new_labels_to_index = labels_to_index.copy()

    X_train, X_val, Y_train, Y_val, new_labels_to_index = filter_by_labels(AvsW, dataset, concatenate_channels,
                                                                           new_labels_to_index, split_by, val_perc)

    if len(X_train.shape) == 2:
        X_train, X_val = np.expand_dims(X_train, axis=2), np.expand_dims(X_val, axis=2)

    return X_train, X_val, Y_train, Y_val, new_labels_to_index


def filter_by_labels(AvsW, dataset, concat_channels, new_labels_to_index, split_by, split_perc):
    win_size = 1000
    nr_trials = 20

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
