import collections

import numpy as np
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint

from WavenetClassifier import classify_params, callbacks
from WavenetClassifier.wavenet_classifier_model import get_wavenet_model
from callbacks.MetricsPlotCallback import MetricsPlotCallback
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from svm.svm import load_cat_tf_record
from utils.plot_utils import create_dir_if_not_exists
from utils.tf_utils import configure_gpu


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


def new_train_test_split(data_dict, movies_to_keep, val_perc, AvsW, labels_to_index, concatenate_channels, seed):
    np.random.seed(seed)
    nr_trials = 20
    shuffled_trial_indexes = np.arange(nr_trials)
    np.random.shuffle(shuffled_trial_indexes)
    nr_val_trials = round(val_perc * nr_trials)
    train_trials = shuffled_trial_indexes[:-nr_val_trials]
    val_trials = shuffled_trial_indexes[-nr_val_trials:]
    print("Validation trials:{}".format(val_trials))
    dataset = [list(data_dict[movie].values()) for movie in movies_to_keep]  # list of 3 values
    new_labels_to_index = labels_to_index.copy()

    X_train, X_val, Y_train, Y_val, new_labels_to_index = filter_by_labels(AvsW, dataset, concatenate_channels,
                                                                           labels_to_index,
                                                                           new_labels_to_index, train_trials,
                                                                           val_trials)

    X_train = np.expand_dims(X_train, axis=2)
    X_val = np.expand_dims(X_val, axis=2)

    return X_train, X_val, Y_train, Y_val, new_labels_to_index


def filter_by_labels(AvsW, dataset, concat_channels, labels_to_index, new_labels_to_index, train_trials, val_trials):
    if AvsW == "all":

        X_train = []
        Y_train = []
        for movie in dataset:
            for trial_index in train_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        X_train.append(example[0].reshape((47, -1)).transpose())
                        Y_train.append([example[1]])
                    else:
                        nr_examples_indiv_channels = example[0].size // 1000
                        for i in range(nr_examples_indiv_channels):
                            X_train.append(example[0][i * 1000: (i + 1) * 1000])
                            Y_train.append(example[1])

        X_val = []
        Y_val = []
        for movie in dataset:
            for trial_index in val_trials:
                for example in movie[trial_index]:
                    if concat_channels:
                        X_val.append(example[0].reshape((47, -1)).transpose())
                        Y_val.append([example[1]])
                    else:
                        nr_examples_indiv_channels = example[0].size // 1000
                        for i in range(nr_examples_indiv_channels):
                            X_val.append(example[0][i * 1000: (i + 1) * 1000])
                            Y_val.append(example[1])

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

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
                        nr_examples_indiv_channels = example[0].size // 1000
                        for i in range(nr_examples_indiv_channels):
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                X_train.append(example[0][i * 1000: (i + 1) * 1000])
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
                        nr_examples_indiv_channels = example[0].size // 1000
                        for i in range(nr_examples_indiv_channels):
                            X_val.append(example[0][i * 1000: (i + 1) * 1000])
                            if example[1] == aerial_view_index or example[1] == water_channel_index:
                                X_train.append(example[0][i * 1000: (i + 1) * 1000])
                                Y_train.append(new_aerial_view_index if
                                               example[1] == aerial_view_index else new_water_channel_index)

        X_train, Y_train, X_val, Y_val = np.stack(X_train), np.stack(Y_train), np.stack(X_val), np.stack(Y_val)

        new_labels_to_index = {b"aerial_view": new_aerial_view_index, b"water_channel": new_water_channel_index}

    elif AvsW == "merge":
        aerial_view_index = labels_to_index[b'aerial_view']
        water_channel_index = labels_to_index[b'water_channel']
        new_index = min(aerial_view_index, water_channel_index)
        max_index = max(aerial_view_index, water_channel_index)
        dataset_train = [frame for movie in dataset for trial_index in train_trials for frame in movie[trial_index]]
        X_train, Y_train = np.array([sample[0] for sample in dataset_train]), \
                           np.array(
                               [sample[1] if sample[1] not in [aerial_view_index, water_channel_index] else new_index
                                for sample in dataset_train])

        dataset_val = [frame for movie in dataset for trial_index in val_trials for frame in movie[trial_index]]
        X_val, Y_val = np.array([sample[0] for sample in dataset_val]), \
                       np.array([sample[1] if sample[1] not in [aerial_view_index, water_channel_index] else
                                 new_index for sample in dataset_val])

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
                                                                               seed)
    print(X_train.max())
    print(X_val.max())

    Y_new = np.concatenate((Y_train.flatten(), Y_val.flatten()))
    class_count = collections.Counter(Y_new)
    nr_samples = Y_new.size

    class_weights = {key: 1 - val / nr_samples if model_parameters["ClassW"] else 1 for key, val in class_count.items()}

    classes = get_classes_list(new_labels_to_index, model_parameters["AvsW"])

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
