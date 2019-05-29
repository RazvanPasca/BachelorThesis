import collections

import numpy as np
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint

from WavenetClassifier import classify_params, callbacks
from WavenetClassifier.wavenet_classifier_model import get_wavenet_model
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from svm.svm import load_cat_tf_record
from utils.plot_utils import create_dir_if_not_exists
from utils.tf_utils import configure_gpu, shuffle_indices


def get_classes_list(labels_to_index, AvsW):
    classes = []
    if AvsW in ["all", "merge"]:
        for index in range(len(labels_to_index)):
            for label, label_index in labels_to_index.items():
                if index == label_index:
                    classes.append(str(label))
    else:
        aerial_index = labels_to_index[b'aerial_view']
        water_index = labels_to_index[b'water_channel']
        classes = ['water_channel', 'aerial_view'] if aerial_index > water_index else ['aerial_view', 'water_channel']

    return classes


def new_train_test_split(data_dict, movies_to_keep, val_perc, AvsW, labels_to_index, concatenate_channels, seed,
                         split_by):
    np.random.seed(seed)

    dataset = [list(data_dict[movie].values()) for movie in movies_to_keep]  # list of 3 values
    new_labels_to_index = labels_to_index.copy()

    X_train, X_val, Y_train, Y_val, new_labels_to_index = filter_by_labels(AvsW, dataset, concatenate_channels,
                                                                           new_labels_to_index, split_by, val_perc)

    return np.expand_dims(X_train, axis=2), np.expand_dims(X_val, axis=2), Y_train, Y_val, new_labels_to_index


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


def get_updated_labels(AvsW, concat_channels, dataset, labels_to_index, new_labels_to_index, train_trials, val_trials,
                       win_size):
    if AvsW == "all":
        pass


    elif AvsW == "1v1":
        aerial_view_index = labels_to_index[b'aerial_view']
        water_channel_index = labels_to_index[b'water_channel']
        new_aerial_view_index, new_water_channel_index = (0, 1) if aerial_view_index < water_channel_index else (1, 0)

        X_train = []
        Y_train = []
        for movie in dataset:
            for trial_index in train_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        if example[1] == aerial_view_index or example[1] == water_channel_index:
                            X_train.append(example[0].reshape((47, -1)).transpose())
                            Y_train.append([new_aerial_view_index if
                                            example[1] == aerial_view_index else new_water_channel_index])
                    else:
                        nr_examples_indiv_channels = example[0].size // win_size
                        for i in range(nr_examples_indiv_channels):
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                X_train.append(example[0][i * win_size: (i + 1) * win_size])
                                Y_train.append([new_aerial_view_index if
                                                example[1] == aerial_view_index else new_water_channel_index])

        X_val = []
        Y_val = []
        for movie in dataset:
            for trial_index in val_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        if example[1] == aerial_view_index or example[1] == water_channel_index:
                            X_train.append(example[0].reshape((47, -1)).transpose())
                            Y_train.append([new_aerial_view_index if
                                            example[1] == aerial_view_index else new_water_channel_index])
                    else:
                        nr_examples_indiv_channels = example[0].size // win_size
                        for i in range(nr_examples_indiv_channels):
                            X_val.append(example[0][i * win_size: (i + 1) * win_size])
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                X_train.append(example[0][i * win_size: (i + 1) * win_size])
                                Y_train.append(new_aerial_view_index if
                                               example[1] == aerial_view_index else new_water_channel_index)

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

        new_labels_to_index = {b"aerial_view": new_aerial_view_index, b"water_channel": new_water_channel_index}

    elif AvsW == "merge":
        aerial_view_index = labels_to_index[b'aerial_view']
        water_channel_index = labels_to_index[b'water_channel']
        new_index = min(aerial_view_index, water_channel_index)
        max_index = max(aerial_view_index, water_channel_index)
        X_train = []
        Y_train = []
        for movie in dataset:
            for trial_index in train_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        X_train.append(example[0].reshape((47, -1)).transpose())
                        if example[1] == aerial_view_index or example[1] == water_channel_index:
                            Y_train.append([new_index])
                        else:
                            Y_train.append([example[1] if example[1] < max_index else example[1] - 1])
                    else:
                        nr_examples_indiv_channels = example[0].size // win_size
                        for i in range(nr_examples_indiv_channels):
                            X_train.append(example[0][i * win_size: (i + 1) * win_size])
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                Y_train.append([new_index])
                            else:
                                Y_train.append([example[1] if example[1] < max_index else example[1] - 1])

        X_val = []
        Y_val = []
        for movie in dataset:
            for trial_index in val_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        X_val.append(example[0].reshape((47, -1)).transpose())
                        if example[1] == aerial_view_index or example[1] == water_channel_index:
                            Y_val.append([new_index])
                        else:
                            Y_val.append([example[1] if example[1] < max_index else example[1] - 1])
                    else:
                        nr_examples_indiv_channels = example[0].size // win_size
                        for i in range(nr_examples_indiv_channels):
                            X_val.append(example[0][i * win_size: (i + 1) * win_size])
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                Y_val.append([new_index])
                            else:
                                Y_val.append([example[1] if example[1] < max_index else example[1] - 1])

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

        del new_labels_to_index[b'aerial_view']
        del new_labels_to_index[b'water_channel']
        new_labels_to_index[b'merged'] = new_index
        new_labels_to_index = {key: val - 1 if val > max_index else val for key, val in new_labels_to_index.items()}
    return X_train, X_val, Y_train, Y_val, new_labels_to_index


def train_test_split(x, y, test_size, shuffle_seed, AvsW, labels_to_index):
    np.random.seed(shuffle_seed)
    new_labels_to_index = labels_to_index

    if AvsW == "1v1":
        aerial_view_index = labels_to_index[b'aerial_view']
        water_channel_index = labels_to_index[b'water_channel']
        keep_indices = np.where((y == aerial_view_index) | (y == water_channel_index))
        x = x[keep_indices]
        y = y[keep_indices]
        new_aerial_view_index = 1 if aerial_view_index > water_channel_index else 0
        new_water_channel_index = 0 if aerial_view_index > water_channel_index else 1
        y[y == aerial_view_index] = new_aerial_view_index
        y[y == water_channel_index] = new_water_channel_index
        new_labels_to_index = {b"aerial_view": new_aerial_view_index, b"water_channel": new_water_channel_index}

    elif AvsW == "merge":
        aerial_view_index = labels_to_index[b'aerial_view']
        water_channel_index = labels_to_index[b'water_channel']
        replace_indices = np.where((y == aerial_view_index) | (y == water_channel_index))
        new_index = min(aerial_view_index, water_channel_index)
        max_index = max(aerial_view_index, water_channel_index)
        y[replace_indices] = new_index
        y[y > max_index] = y[y > max_index] - 1
        del new_labels_to_index[b'aerial_view']
        del new_labels_to_index[b'water_channel']
        new_labels_to_index[b'merged'] = new_index
        new_labels_to_index = {key: val - 1 if val > max_index else val for key, val in new_labels_to_index.items()}

    nr_train_examples = round((1 - test_size) * y.size)
    indices = np.arange(y.size)
    np.random.shuffle(indices)

    x_shuffled = x[indices]
    y_shuffled = y[indices]
    return x_shuffled[:nr_train_examples], x_shuffled[nr_train_examples:], \
           y_shuffled[:nr_train_examples], y_shuffled[nr_train_examples:], new_labels_to_index


def train_model(model_params, X_train, Y_train, X_test, Y_test, classes, model_path, class_weights):
    model = get_wavenet_model(nr_filters=model_params["nr_filters"],
                              input_shape=(X_train.shape[1:]),
                              nr_layers=model_params["nr_layers"],
                              lr=model_params["lr"],
                              clipvalue=model_params["clip_grad_by_value"],
                              skip_conn_filters=model_params["skip_conn_filters"],
                              regularization_coef=model_params["regularization_coef"],
                              nr_output_classes=len(classes))

    tboard_callback = TensorBoard(log_dir=model_path, write_graph=True)
    log_callback = CSVLogger(model_path + "/session_log.csv")
    plot_metric_callback = MetricsPlotCallback(model_path)
    conf_matrix_callback = callbacks.ConfusionMatrixPlotter(X_train, Y_train,
                                                            X_test, Y_test,
                                                            classes,
                                                            model_path,
                                                            model_params["logging_period"],
                                                            normalize=True)
    save_model_callback = ModelCheckpoint(filepath="{}/best_model.h5".format(model_path),
                                          monitor="val_loss",
                                          save_best_only=True)
    print(model.summary())
    model.fit(x=X_train,
              y=Y_train,
              batch_size=model_params["batch_size"],
              epochs=model_params["n_epochs"],
              validation_data=(X_test, Y_test),
              verbose=2,
              shuffle=True,
              class_weight=class_weights,
              callbacks=[tboard_callback, log_callback, save_model_callback, plot_metric_callback,
                         conf_matrix_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


def main(movies_to_keep, val_perc, concatenate_channels, seed):
    data_dict, labels_to_index = load_cat_tf_record(
        CAT_TFRECORDS_PATH_TOBEFORMATED.format(model_parameters["window_size"]), model_parameters["cutoff_freq"])

    X_train, X_val, Y_train, Y_val, new_labels_to_index = new_train_test_split(data_dict, movies_to_keep, val_perc,
                                                                               model_parameters["AvsW"],
                                                                               labels_to_index, concatenate_channels,
                                                                               seed,
                                                                               model_parameters["split_by"])
    print(X_train.max())
    print(X_val.max())

    train_counter = collections.Counter(Y_train.flatten())
    val_counter = collections.Counter(Y_val.flatten())
    class_count = train_counter + val_counter
    nr_samples = Y_train.size + Y_val.size

    class_weights = {key: 1 - val / nr_samples if model_parameters["ClassW"] else 1 for key, val in class_count.items()}

    classes = get_classes_list(new_labels_to_index, model_parameters["AvsW"])

    print(train_counter)
    print(val_counter)
    print(class_count)
    print(new_labels_to_index)
    print(class_weights)
    print(X_train.shape)
    print(Y_train.shape)
    print(len(classes))

    train_model(model_parameters, X_train, Y_train, X_val, Y_val, classes, model_path, class_weights)


if __name__ == '__main__':
    model_parameters = classify_params.model_params
    configure_gpu(model_parameters["gpu"])
    model_path = classify_params.get_model_name(model_parameters)
    create_dir_if_not_exists(model_path)
    file_redirect = "{}/output.txt"
    # if not (file_redirect is None):
    #     sys.stdout = open(file_redirect.format(model_path), 'w+')

    main(movies_to_keep=model_parameters["movies_to_keep"],
         val_perc=model_parameters["val_perc"],
         concatenate_channels=model_parameters["concatenate_channels"],
         seed=model_parameters["shuffle_seed"])
