import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks
from keras.callbacks import TensorBoard


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def prepare_file_for_writing(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


def decode_model_output(model_logits, model_params):
    if model_params.get_classifying() == 2:
        if model_params.output == "Regression":
            return model_logits[0][0]
        else:
            return get_bin_output(model_logits[1], model_params)
    elif model_params.get_classifying() == 1:
        return get_bin_output(model_logits, model_params)
    else:
        return model_logits


def get_bin_output(model_logits, model_params):
    bin_index = np.argmax(model_logits)
    a = (model_params.dataset.bins[bin_index - 1] + model_params.dataset.bins[bin_index]) / 2
    return a


def plot_predictions(original_sequence, image_title, nr_predictions, frame_size, predicted_sequence,
                     save_path,
                     starting_point,
                     prediction_losses=None,
                     vlines_coords=None):
    title = image_title
    x1 = range(starting_point + frame_size - 5, starting_point + nr_predictions + frame_size)
    x2 = range(starting_point + frame_size, starting_point + nr_predictions + frame_size)
    y1 = original_sequence[x1]
    label1 = "Original sequence"
    label2 = "Predicted sequence"
    # for _ in range(5):
    #     prediction_losses = np.insert(prediction_losses, 0, np.nan)
    plot_2_overlapped_series(x1, y1, label1, x2, predicted_sequence, label2, title, save_path, prediction_losses,
                             vlines_coords)


def plot_2_overlapped_series(x1, y1, label1, x2, y2, label2, image_title, save_path, prediction_losses=None,
                             vlines_coords=None):
    plt.figure(figsize=(16, 12))

    nr_plot_rows = 2 if prediction_losses is not None else 1
    plt.subplot(nr_plot_rows, 1, 1)
    plt.title(image_title)
    plt.plot(x1, y1, '.-', label=label1, color="blue")
    plt.plot(x2, y2, '.-', label=label2, color="red")

    l, r = plt.xlim()
    low_lim = np.min(y1)
    hi_lim = np.max(y1)

    low_lim = low_lim - abs(low_lim / 2)
    hi_lim = hi_lim + abs(hi_lim / 2)
    plt.ylim(low_lim, hi_lim)
    ymin = min(low_lim, np.min(y2))
    ymax = max(hi_lim, np.max(y2))
    if vlines_coords is not None:
        plt.vlines(vlines_coords, ymin=ymin, ymax=ymax, lw=0.2)
    plt.legend()

    if prediction_losses is not None:
        plt.subplot(nr_plot_rows, 1, 2)
        plt.plot(x2, prediction_losses)
        plt.xlim(l, r)
        ymin = np.nanmin(prediction_losses)
        ymax = np.nanmax(prediction_losses)

        if vlines_coords is not None:
            plt.vlines(vlines_coords[:-1], ymin=ymin, ymax=ymax, lw=0.2)

    plt.savefig(save_path + '/' + image_title + ".png")
    plt.close()


def get_channel_index_from_name(name):
    index = name.find("C:")
    index_ = index + 2
    c = name[index_]

    while c.isdigit():
        index_ += 1
        c = name[index_]

    return int(name[index + 2:index_])


def get_predictions_with_losses(model, model_params, original_sequence, nr_predictions, image_name, starting_point,
                                generated_window_size, plot=True):
    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    predictions_losses = np.zeros(nr_actual_predictions)
    vlines_coords = []

    if generated_window_size > nr_actual_predictions:
        warnings.warn(
            "Can't generate more steps per window than the total number of predictions. "
            "The generated window size will be set to the total number of predictions ")
        generated_window_size = nr_actual_predictions

    input_sequence = []
    for prediction_nr in range(nr_actual_predictions):
        if prediction_nr % generated_window_size == 0:
            input_sequence = np.reshape(
                original_sequence[
                starting_point + prediction_nr:starting_point + prediction_nr + model_params.frame_size],
                (-1, model_params.frame_size, 1))
            vlines_coords.append(starting_point + prediction_nr + model_params.frame_size - 1)

        # channel_index = get_channel_index_from_name(image_name)

        predicted = decode_model_output(model.predict(input_sequence), model_params)
        predicted_sequence[prediction_nr] = predicted
        input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)

        curr_loss = np.abs(predicted - original_sequence[starting_point + prediction_nr + model_params.frame_size])
        predictions_losses[prediction_nr] = curr_loss if prediction_nr % generated_window_size == 0 else \
            predictions_losses[prediction_nr - 1] + curr_loss

    if plot:
        image_name += "GenSteps:{}_".format(generated_window_size)
        plot_predictions(original_sequence, image_name, nr_actual_predictions,
                         model_params.frame_size, predicted_sequence, model_params.model_path, starting_point,
                         prediction_losses=predictions_losses,
                         vlines_coords=vlines_coords)

    return predictions_losses, predicted_sequence


def generate_prediction_name(seq_addr):
    name = ''
    for key in seq_addr:
        if key == "SOURCE":
            pass
        else:
            name += '{}:{}_'.format(key, str(seq_addr[key]))
    return name


def generate_subplots(original_sequences, sequence_predictions, sequence_names, prediction_starting_point, save_path):
    nr_rows = 2
    nr_cols = len(sequence_predictions) // nr_rows
    prediction_length = len(sequence_predictions[0])
    fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(20, 10))
    predictions_x_indices = range(prediction_starting_point, prediction_starting_point + prediction_length)
    original_sequences_x_indices = range(len(original_sequences[0]))

    for i, subplot in enumerate(subplots.flatten()):
        subplot.plot(original_sequences_x_indices, original_sequences[i][original_sequences_x_indices], '.-',
                     label="Original sequence", color="blue")
        subplot.plot(predictions_x_indices, sequence_predictions[i], '.-', label="Predicted sequence", color="red")
        subplot.set_title(sequence_names[i])
        subplot.legend()

    plt.tight_layout()
    plt.savefig(save_path, format="png")
    # plt.show()
    plt.close()


def get_predictions(model, model_params, epoch, starting_point, nr_prediction_steps):
    pred_seqs = model_params.dataset.prediction_sequences

    for source in pred_seqs:
        dir_name = "{}/{}".format(model_params.model_path, source)
        create_dir_if_not_exists(dir_name)

        for generated_window_size in [1, nr_prediction_steps]:
            plt_path = "{}/E:{}_GenWindowSize:{}".format(dir_name, epoch, generated_window_size)
            sequence_predictions = []
            sequence_names = []
            original_sequences = []

            for sequence, addr in pred_seqs[source]:
                image_prefix = generate_prediction_name(addr)
                image_prefix = addr["SOURCE"] + "_" + image_prefix
                image_name = "{}_E:{}_GenWinSize:{}".format(image_prefix, epoch, generated_window_size)

                _, predictions = get_predictions_with_losses(model, model_params, sequence, nr_prediction_steps,
                                                             image_name,
                                                             starting_point,
                                                             generated_window_size, plot=False)

                sequence_predictions.append(predictions)
                sequence_names.append(image_name)
                original_sequences.append(sequence)

            generate_subplots(original_sequences, sequence_predictions, sequence_names,
                              starting_point + model_params.frame_size, plt_path)


class PlotCallback(callbacks.Callback):
    """Callback used for plotting at certain epochs the generations of the model

        If nr_predictions is smaller than 1, the model will predict up to the end of the signal

        If starting point is smaller than 1, the model will start predicting from the beginning of the signal-frame size
    """

    def __init__(self, model_params, plot_period, nr_predictions, starting_point):
        super().__init__()

        self.model_params = model_params
        self.epoch = 0
        self.nr_prediction_steps = nr_predictions if nr_predictions > 0 else model_params.dataset.trial_length
        self.plot_period = plot_period
        self.starting_point = starting_point

    def on_train_begin(self, logs={}):
        create_dir_if_not_exists(self.model_params.model_path)
        self.model_params.serialize_to_json(self.model_params.model_path)
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

        if self.epoch % self.plot_period == 0 or self.epoch == 1:
            get_predictions(self.model, self.model_params, self.epoch, self.starting_point,
                            nr_prediction_steps=self.nr_prediction_steps)

    def on_train_end(self, logs=None):
        get_predictions(self.model, self.model_params, "TrainEnd", self.starting_point,
                        nr_prediction_steps=self.nr_prediction_steps)


class TensorBoardWrapper(TensorBoard):
    """Sets the self.validation_data property for use with TensorBoard callback."""

    def __init__(self, batch_gen, nb_steps, **kwargs):
        super().__init__(**kwargs)
        self.batch_gen = batch_gen  # The generator.
        self.nb_steps = nb_steps  # Number of times to call next() on the generator.

    def on_epoch_end(self, epoch, logs=None):
        # Fill in the `validation_data` property. Obviously this is specific to how your generator works.
        # Below is an example that yields images and classification tags.
        # After it's filled in, the regular on_epoch_end method has access to the validation_data.
        x, y = next(self.batch_gen)
        self.validation_data = (x, y, np.ones(self.batch_size))
        return super().on_epoch_end(epoch, logs)
