import sys

import numpy as np
from keras.callbacks import CSVLogger, TensorBoard, ModelCheckpoint

from CNN_class import classifier_training_params
from CNN_class.wavenet_classifier import get_wavenet_model
from datasets.paths import CAT_TFRECORDS_PATH_TOBEFORMATED
from plot_utils import create_dir_if_not_exists
from svm.svm import load_cat_tf_record
from tf_utils import configure_gpu


def train_test_split(x, y, test_size, shuffle_seed):
    np.random.seed(shuffle_seed)
    nr_train_examples = round(test_size * y.size)
    indices = np.arange(y.size)
    np.random.shuffle(indices)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    return x_shuffled[:nr_train_examples], x_shuffled[nr_train_examples:], \
           y_shuffled[:nr_train_examples], y_shuffled[nr_train_examples:]


def train_model(model_params, X_train, Y_train, X_test, Y_test, nr_classes, model_path):
    model = get_wavenet_model(nr_filters=model_params["nr_filters"],
                              input_shape=(X_train.shape[1:]),
                              nr_layers=model_params["nr_layers"],
                              lr=model_params["lr"],
                              clipvalue=model_params["clip_grad_by_value"],
                              skip_conn_filters=model_params["skip_conn_filters"],
                              regularization_coef=model_params["regularization_coef"],
                              nr_output_classes=nr_classes,
                              multiloss_weights=model_params["multiloss_weights"])

    tensorboard_callback = TensorBoard(log_dir=model_path, write_graph=True)
    log_callback = CSVLogger(model_path + "/session_log.csv")

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
              callbacks=[tensorboard_callback, log_callback, save_model_callback])

    print('Saving model and results...')
    model.save(model_params.model_path + "/" + "final_model.h5")
    print('\nDone!')


if __name__ == '__main__':
    model_parameters = classifier_training_params.model_params
    configure_gpu(model_parameters["gpu"])
    file_redirect = "{}/output_windowsize_{}.txt"

    model_path = classifier_training_params.get_model_name(model_parameters)
    create_dir_if_not_exists(model_path)
    for window_size in [1000]:
        if not (file_redirect is None):
            sys.stdout = open(file_redirect.format(model_path, window_size), 'w+')

        X, Y, labels_to_index = load_cat_tf_record(CAT_TFRECORDS_PATH_TOBEFORMATED.format(window_size),
                                                   model_parameters["cutoff_freq"])
        X = np.expand_dims(X, axis=2)
        print(labels_to_index)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                            test_size=model_parameters["val_coverage_per_epoch"],
                                                            shuffle_seed=42)
        train_model(model_parameters, X_train, Y_train, X_test, Y_test, len(labels_to_index), model_path)
