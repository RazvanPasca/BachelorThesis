import collections

import numpy as np
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint

import callbacks.MetricsPlotCallback
from Wavenet_class import classify_params, callbacks
from Wavenet_class.wavenet_classifier_model import get_wavenet_model
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from plot_utils import create_dir_if_not_exists
from svm.svm import load_cat_tf_record
from tf_utils import configure_gpu


def get_classes_list(labels_to_index, AvsW):
    classes = []
    if AvsW in ["1v1", "merge"]:
        for index in range(len(labels_to_index)):
            for label, label_index in labels_to_index.items():
                if index == label_index:
                    classes.append(str(label))
    else:
        aerial_index = labels_to_index[b'aerial_view']
        water_index = labels_to_index[b'water_channel']
        classes = ['water_channel', 'aerial_view'] if aerial_index > water_index else ['aerial_view', 'water_channel']

    return classes


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
    metric_callback = callbacks.MetricsPlotCallback.MetricsPlotCallback(model_path)
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
              callbacks=[tboard_callback, log_callback, save_model_callback, metric_callback, conf_matrix_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


def main():
    X, Y, labels_to_index = load_cat_tf_record(
        CAT_TFRECORDS_PATH_TOBEFORMATED.format(model_parameters["window_size"]), model_parameters["cutoff_freq"])

    X = np.expand_dims(X, axis=2)
    print(labels_to_index)

    X_train, X_test, Y_train, Y_test, new_labels_to_index = train_test_split(X, Y,
                                                                             test_size=model_parameters[
                                                                                 "val_coverage_per_epoch"],
                                                                             shuffle_seed=model_parameters[
                                                                                 "shuffle_seed"],
                                                                             AvsW=model_parameters["AvsW"],
                                                                             labels_to_index=labels_to_index)

    print(new_labels_to_index)

    Y_new = np.concatenate((Y_train, Y_test))
    class_count = collections.Counter(Y_new)
    nr_samples = Y_new.size
    print(class_count)

    class_weights = {key: 1 - val / nr_samples if model_parameters["ClassW"] else 1 for key, val in class_count.items()}
    classes = get_classes_list(new_labels_to_index, model_parameters["AvsW"])
    train_model(model_parameters, X_train, Y_train, X_test, Y_test, classes, model_path, class_weights)


if __name__ == '__main__':
    model_parameters = classify_params.model_params
    configure_gpu(model_parameters["gpu"])
    model_path = classify_params.get_model_name(model_parameters)
    create_dir_if_not_exists(model_path)
    file_redirect = "{}/output.txt"
    # if not (file_redirect is None):
    #     sys.stdout = open(file_redirect.format(model_path), 'w+')

    main()
