import os

import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def decode_model_output(model_logits, model_params):
    if model_params.get_classifying():
        bin_index = np.argmax(model_logits)
        a = (model_params.dataset.bins[bin_index - 1] + model_params.dataset.bins[bin_index]) / 2
        return a
    return model_logits


def plot_predictions(original_sequence, image_title, nr_predictions, frame_size, predicted_sequence,
                     save_path,
                     starting_point,
                     teacher_forcing,
                     prediction_losses=None,
                     vlines_coords=None):
    title = image_title + "TF:{}".format(teacher_forcing)
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
    ymin = min(np.min(y1), np.min(y2))
    ymax = max(np.max(y1), np.max(y2))
    if vlines_coords is not None:
        plt.vlines(vlines_coords, ymin=ymin, ymax=ymax, lw=0.1)
    plt.legend()

    if prediction_losses is not None:
        plt.subplot(nr_plot_rows, 1, 2)
        plt.plot(x2, prediction_losses)
        plt.xlim(l, r)
        ymin = np.nanmin(prediction_losses)
        ymax = np.nanmax(prediction_losses)
        if vlines_coords is not None:
            plt.vlines(vlines_coords[:-1], ymin=ymin, ymax=ymax, lw=0.1)

    plt.savefig(save_path + '/' + image_title + ".png")
    plt.close()


def get_predictions_on_sequence(model,
                                model_params,
                                original_sequence,
                                nr_predictions,
                                image_name,
                                starting_point=0,
                                teacher_forcing=True):
    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    position = 0

    if teacher_forcing:
        for step in range(starting_point, starting_point + nr_actual_predictions):
            input_sequence = np.reshape(original_sequence[step:step + model_params.frame_size],
                                        (-1, model_params.frame_size, 1))
            predicted = decode_model_output(model.predict(input_sequence), model_params)
            predicted_sequence[position] = predicted
            position += 1

    else:
        input_sequence = np.reshape(original_sequence[starting_point:starting_point + model_params.frame_size],
                                    (-1, model_params.frame_size, 1))
        for step in range(starting_point, starting_point + nr_actual_predictions):
            predicted = decode_model_output(model.predict(input_sequence), model_params)
            predicted_sequence[position] = predicted
            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
            position += 1

    plot_predictions(original_sequence, image_name, nr_actual_predictions,
                     model_params.frame_size, predicted_sequence, model_params.model_path, starting_point,
                     teacher_forcing)


def get_predictions_with_losses(model, model_params, original_sequence, nr_predictions, image_name, starting_point,
                                generated_window_size, plot=True):

    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    predictions_losses = np.zeros(nr_actual_predictions)
    vlines_coords = []

    if generated_window_size > nr_actual_predictions:
        raise ValueError("Can't generate more steps per slice than the number of predictions")

    input_sequence = []
    for prediction_nr in range(nr_actual_predictions):
        if prediction_nr % generated_window_size == 0:
            input_sequence = np.reshape(
                original_sequence[
                starting_point + prediction_nr:starting_point + prediction_nr + model_params.frame_size],
                (-1, model_params.frame_size, 1))
            vlines_coords.append(starting_point + prediction_nr + model_params.frame_size - 1)

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
                         teacher_forcing="Partial", prediction_losses=predictions_losses,
                         vlines_coords=vlines_coords)

    return predictions_losses


def generate_prediction_name(seq_addr):
    name = ''
    for key in seq_addr:
        if key == "SOURCE":
            pass
        else:
            name += '{}:{}_'.format(key, str(seq_addr[key]))
    return name


def get_predictions(model, model_params, epoch, starting_point, nr_prediction_steps):
    pred_seqs = model_params.dataset.prediction_sequences
    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            image_name = "/" + addr["SOURCE"] + "/" + "E:{}_".format(epoch) + image_name
            create_dir_if_not_exists(model_params.model_path + "/" + addr["SOURCE"])
            get_predictions_on_sequence(model, model_params, sequence, nr_prediction_steps, image_name, starting_point,
                                        True)
            get_predictions_on_sequence(model, model_params, sequence, nr_prediction_steps, image_name, starting_point,
                                        False)


class PlotCallback(callbacks.Callback):
    def __init__(self, model_params, plot_period, nr_predictions_steps, starting_point):
        super().__init__()
        self.model_params = model_params
        self.epoch = 0
        self.nr_prediction_steps = nr_predictions_steps
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
