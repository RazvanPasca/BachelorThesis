import csv
import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from output_utils import decode_model_output, get_normalized_erros_per_sequence


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def prepare_file_for_writing(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


def get_channel_index_from_name(name):
    index = name.find("C:")
    index_ = index + 2
    c = name[index_]

    while c.isdigit():
        index_ += 1
        c = name[index_]

    return int(name[index + 2:index_])


def generate_prediction_name(seq_addr):
    name = ''
    for key in seq_addr:
        if key == "SOURCE":
            pass
        else:
            name += '{}:{}_'.format(key, str(seq_addr[key]))
    return name


def generate_multi_plot(model, model_params, epoch, starting_point, nr_prediction_steps, all_reset_indices,
                        nr_of_sequences_to_plot, nr_rows):
    pred_seqs = model_params.dataset.prediction_sequences
    all_prediction_losses_norm_mean = {}

    for source in pred_seqs:
        dir_name_root = "{}/{}".format(model_params.model_path, source)
        all_prediction_losses_norm_mean[source] = []

        for reset_indices in all_reset_indices:
            reset_indices_, reset_indices_np_sorted = get_reset_indices_array(model_params, reset_indices)

            dir_name = "{}/WinSize:{}".format(dir_name_root, reset_indices_[:10])
            create_dir_if_not_exists(dir_name)
            plt_path = "{}/E:{}_Conditioning:{}".format(dir_name, epoch, model_params.condition_on_gamma)

            sequence_predictions = []
            sequence_names = []
            original_sequences = []
            vlines_coords_list = []
            prediction_losses_normalized = np.zeros(reset_indices_.size)
            prediction_losses = []

            for sequence, addr in pred_seqs[source][:nr_of_sequences_to_plot]:
                image_prefix = generate_prediction_name(addr)
                image_prefix = addr["SOURCE"] + "_" + image_prefix
                image_name = "{}_E:{}".format(image_prefix, epoch)

                losses, predictions, vlines_coords = get_sequence_prediction(model, model_params, sequence,
                                                                             nr_prediction_steps,
                                                                             image_name,
                                                                             starting_point,
                                                                             reset_indices, plot=False)

                normalized_errors_per_sequence = get_normalized_erros_per_sequence(losses[0], reset_indices_)
                prediction_losses_normalized += normalized_errors_per_sequence
                prediction_losses.append(losses)
                sequence_predictions.append(predictions)
                sequence_names.append(image_name)
                original_sequences.append(sequence)
                vlines_coords_list.append(vlines_coords)

            generate_subplots(original_sequences, sequence_predictions, vlines_coords_list, sequence_names,
                              starting_point + model_params.frame_size, plt_path,
                              model_params.gamma_windows_in_trial, nr_rows)
            generate_errors_subplots(prediction_losses, vlines_coords_list, sequence_names,
                                     starting_point + model_params.frame_size, plt_path,
                                     model_params.gamma_windows_in_trial, nr_rows)

            prediction_losses_normalized_seq_avg = prediction_losses_normalized / nr_of_sequences_to_plot
            log_range_accumulated_errors(epoch, prediction_losses_normalized_seq_avg, reset_indices_, dir_name)
            all_prediction_losses_norm_mean[source].append(np.mean(prediction_losses_normalized_seq_avg))

    return all_prediction_losses_norm_mean


def get_reset_indices_array(model_params, reset_indices):
    reset_indices_np_sorted = (np.array(list(reset_indices)) - model_params.frame_size)
    reset_indices_np_sorted.sort()
    pos_reset_indices = reset_indices_np_sorted > 0
    reset_indices_ = reset_indices_np_sorted[pos_reset_indices]
    return reset_indices_, reset_indices_np_sorted


def log_range_accumulated_errors(epoch, prediction_losses, reset_indices, dir_name):
    csv_name = os.path.join(dir_name, "prediction_losses_means.csv")

    if os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            errors_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            errors_writer.writerow(np.insert(prediction_losses, 0, epoch))

    else:
        with open(csv_name, 'w') as f:
            errors_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            errors_writer.writerow(["Epoch"] + list(reset_indices))
            errors_writer.writerow(np.insert(prediction_losses, 0, epoch))


def get_sequence_prediction(model, model_params, original_sequence, nr_predictions, image_name,
                            starting_point,
                            reset_indices, plot=True):
    """
    Generates prediction given an original sequence as input

     Args:
            model: instance of the keras model used for predicting
            model_params: instance of the model_params associated with this training/testing session
            original_sequence: the original sequence from the dataset used as "seed" for the model
            nr_predictions: how many values we will predict in total
            starting_point: the place from which we start seeding the model with model_params.frame_size values
            reset_indices: when we reset the teacher forcing, starting with another original sequence
            plot: boolean telling if we should plot or not the results or only return them

    Returns:
            prediction_losses: the accumulated errors associated with it
            predicted_sequences: the predicted sequence
            vlines_coords: a vlines object indicating the reset indices
    """
    predicted_sequences = [np.zeros(nr_predictions) for _ in range(model_params.get_classifying())]
    predictions_losses = [np.zeros(nr_predictions) for _ in range(model_params.get_classifying())]
    vlines_coords = []
    input_sequence = []

    for prediction_nr in range(nr_predictions):
        current_pos = starting_point + prediction_nr

        if current_pos + model_params.frame_size in reset_indices or prediction_nr == 0:
            seed_sequence = original_sequence[current_pos:current_pos + model_params.frame_size]
            input_sequence = np.reshape(seed_sequence, (-1, model_params.frame_size, 2))
            vlines_coords.append(current_pos + model_params.frame_size - 1)

        """I get the predictions here. I might have 2 predictions made if I am using regression and softmax"""
        predicted_values = decode_model_output(model.predict(input_sequence), model_params)
        for i, predicted_val in enumerate(predicted_values):
            predicted_sequences[i][prediction_nr] = predicted_val
            predicted_val_w_label = np.array(
                [predicted_val, original_sequence[current_pos + model_params.frame_size, 1]])

            input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted_val_w_label, (-1, 1, 2)), axis=1)
            actual_value = original_sequence[current_pos + model_params.frame_size][0]
            curr_loss = np.abs(actual_value - predicted_val)
            predictions_losses[i][prediction_nr] = curr_loss if current_pos + model_params.frame_size in reset_indices \
                else predictions_losses[i][prediction_nr - 1] + curr_loss

    return predictions_losses, predicted_sequences, vlines_coords


def generate_subplots(original_sequences, sequence_predictions, vlines_coords_list, sequence_names,
                      prediction_starting_point, save_path, gamma_windows, nr_rows):
    nr_cols = len(sequence_predictions) // nr_rows
    colors = ["red", "green"]
    prediction_length = len(sequence_predictions[0][0])
    fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(25, 15))
    predictions_x_indices = range(prediction_starting_point, prediction_starting_point + prediction_length)
    original_sequences_x_indices = range(len(original_sequences[0]))

    show_vlines = len(vlines_coords_list[0]) != sequence_predictions[0][0].size

    for i, subplot in enumerate(subplots.flatten()):
        sequence_values = original_sequences[i][:, 0]
        subplot.plot(original_sequences_x_indices, sequence_values[original_sequences_x_indices],
                     label="Original sequence", color="blue", linewidth=1)

        for gamma_range in gamma_windows:
            subplot.axvspan(gamma_range[0], gamma_range[1], alpha=0.5)

        for j, sequence_prediction in enumerate(sequence_predictions[i]):
            subplot.plot(predictions_x_indices, sequence_prediction, label="Predicted sequence {}".format(j),
                         color=colors[j], linewidth=1)
        if show_vlines:
            lim = max(np.max(sequence_predictions), np.max(sequence_values))
            lim1 = min(np.min(sequence_predictions), np.min(sequence_values))
            subplot.vlines(vlines_coords_list[i], ymin=lim1, ymax=lim, lw=0.5)
        subplot.set_title(sequence_names[i])
        subplot.legend()

    plt.tight_layout()
    plt.savefig("{}.png".format(save_path), format="png")
    plt.close()


def generate_errors_subplots(prediction_losses, vlines_coords_list, sequence_names, prediction_starting_point,
                             save_path, gamma_windows, nr_rows):
    nr_cols = len(prediction_losses) // nr_rows
    fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(25, 15))
    vlines_coords_list = [np.array(coords_list) - prediction_starting_point + 1 for coords_list in vlines_coords_list]
    show_vlines = vlines_coords_list[0].size != prediction_losses[0][0].size

    if show_vlines:
        lim = np.max(prediction_losses[:nr_cols * nr_rows])
        lim1 = np.min(prediction_losses[:nr_cols * nr_rows])

    for i, subplot in enumerate(subplots.flatten()):
        subplot.plot(prediction_losses[i][0], linewidth=1, label="Prediction accumulated errors", color="blue")
        for gamma_range in gamma_windows:
            subplot.axvspan(gamma_range[0] - prediction_starting_point + 1,
                            gamma_range[1] - prediction_starting_point + 1,
                            alpha=0.5)
        if show_vlines:
            subplot.vlines(vlines_coords_list[i], ymin=lim1, ymax=lim, lw=0.5)
        subplot.set_title(sequence_names[i])
        subplot.legend()

    plt.tight_layout()
    plt.savefig("{}_Accumulated_Errors.png".format(save_path), format="png")
    plt.close()


def plot_conf_matrix(cnf_mat, classes, cmap, normalize, save_path):
    plt.figure(figsize=(16, 12))
    if normalize:
        cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]
    thresh = cnf_mat.max() / 2.
    for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
        plt.text(j, i, "{0:.4f}".format(cnf_mat[i, j]),
                 horizontalalignment="center",
                 color="orange" if cnf_mat[i, j] > thresh else "black")

    plt.imshow(cnf_mat, interpolation='nearest', cmap=cmap)
    # Labels
    if classes is not None:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    plt.colorbar()
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.close()


def plot_pred_losses(pred_losses, name):
    for source, source_pred_errors in pred_losses.items():
        pass
