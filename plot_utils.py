import os

import matplotlib.pyplot as plt
import numpy as np
from keras import callbacks


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def decode_model_output(model_logits, classifying, bins):
    if classifying:
        bin_index = np.argmax(model_logits)
        a = (bins[bin_index - 1] + bins[bin_index]) / 2
        return a
    return model_logits


def plot_predictions(original_sequence, image_title, nr_predictions, frame_size, predicted_sequence, save_path,
                     starting_point,
                     teacher_forcing, vlines_coords=None):
    title = image_title + "TF:{}".format(teacher_forcing)
    x1 = range(starting_point + frame_size, starting_point + nr_predictions + frame_size)
    y1 = original_sequence[starting_point + frame_size:starting_point + nr_predictions + frame_size]
    label1 = "Original sequence"
    label2 = "Predicted sequence"
    plot_2_overlapped_series(x1, y1, label1, x1, predicted_sequence, label2, title, save_path, vlines_coords)


def plot_2_overlapped_series(x1, y1, label1, x2, y2, label2, image_title, save_path, vlines_coords=None):
    plt.figure(figsize=(16, 12))
    plt.title(image_title)
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    ymin = min(np.min(y1), np.min(y2))
    ymax = max(np.max(y1), np.max(y2))
    if vlines_coords is not None:
        plt.vlines(vlines_coords, ymin=ymin, ymax=ymax)
    plt.legend()
    plt.savefig(save_path + '/' + image_title + ".png")
    # plt.show()
    plt.close()


def get_predictions_on_sequence(model,
                                model_params,
                                original_sequence,
                                nr_predictions,
                                image_name,
                                starting_point=0,
                                teacher_forcing=True,
                                nr_generated_steps=1):
    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    position = 0

    nr_generated_steps = min(nr_generated_steps, nr_actual_predictions)

    if teacher_forcing:
        for step in range(starting_point, starting_point + nr_actual_predictions):
            input_sequence = np.reshape(original_sequence[step:step + model_params.frame_size],
                                        (-1, model_params.frame_size, 1))
            predicted = decode_model_output(model.predict(input_sequence), model_params.get_classifying(),
                                            model_params.dataset.bins)
            predicted_sequence[position] = predicted
            position += 1

    else:
        input_sequence = np.reshape(original_sequence[:model_params.frame_size], (-1, model_params.frame_size, 1))
        for step in range(starting_point, starting_point + nr_actual_predictions):
            predicted = decode_model_output(model.predict(input_sequence), model_params.get_classifying(),
                                            model_params.dataset.bins)
            predicted_sequence[position] = predicted
            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
            position += 1

    plot_predictions(original_sequence, image_name, nr_actual_predictions,
                     model_params.frame_size, predicted_sequence, model_params.get_save_path(), starting_point,
                     teacher_forcing)


def get_partial_generated_sequences(model,
                                    model_params,
                                    original_sequence,
                                    nr_predictions,
                                    image_name,
                                    starting_point,
                                    nr_generated_steps):
    nr_actual_predictions = min(nr_predictions, original_sequence.size - starting_point - model_params.frame_size - 1)
    predicted_sequence = np.zeros(nr_actual_predictions)
    position = 0

    nr_generated_steps = min(nr_generated_steps, nr_actual_predictions)

    vlines_coords = []

    while position + nr_generated_steps < nr_actual_predictions:

        input_sequence = np.reshape(
            original_sequence[starting_point + position:starting_point + position + model_params.frame_size],
            (-1, model_params.frame_size, 1))
        vlines_coords.append(starting_point + position + model_params.frame_size)

        for step in range(nr_generated_steps):
            predicted = decode_model_output(model.predict(input_sequence), model_params.get_classifying(),
                                            model_params.dataset.bins)
            predicted_sequence[position] = predicted
            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted, (-1, 1, 1)), axis=1)
            position += 1

    image_name += "GenSteps:{}".format(nr_generated_steps)
    plot_predictions(original_sequence, image_name, nr_actual_predictions,
                     model_params.frame_size, predicted_sequence, model_params.get_save_path(), starting_point,
                     teacher_forcing="Partial", vlines_coords=vlines_coords)


def generate_prediction_name(seq_addr):
    name = ''
    for key in seq_addr:
        name += '{}:{}_'.format(key, str(seq_addr[key]))
    return name


def get_predictions(model, model_params, epoch, starting_point, nr_prediction_steps, nr_generated_steps=1):
    pred_seqs = model_params.dataset.prediction_sequences
    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            image_name = "E:{}_".format(epoch) + image_name
            get_predictions_on_sequence(model, model_params, sequence, nr_prediction_steps, image_name, starting_point,
                                        True, nr_generated_steps)
            get_predictions_on_sequence(model, model_params, sequence, nr_prediction_steps, image_name, starting_point,
                                        False, nr_generated_steps)


class PlotCallback(callbacks.Callback):
    def __init__(self, model_params, plot_period, nr_predictions_steps, starting_point):
        super().__init__()
        self.model_params = model_params
        self.epoch = 0
        self.nr_prediction_steps = nr_predictions_steps
        self.plot_period = plot_period
        self.starting_point = starting_point

    def on_train_begin(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

        if self.epoch % self.plot_period == 0 or self.epoch == 1:
            get_predictions(self.model, self.model_params, self.epoch, self.starting_point,
                            nr_prediction_steps=self.nr_prediction_steps)

    def on_train_end(self, logs=None):
        get_predictions(self.model, self.model_params, "TrainEnd", self.starting_point,
                        nr_prediction_steps=self.nr_prediction_steps)
