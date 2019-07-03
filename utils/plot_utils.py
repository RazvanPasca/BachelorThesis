import itertools

import numpy as np
from matplotlib import pyplot as plt

from utils.output_utils import get_normalized_pred_errors, get_sequence_prediction
from utils.system_utils import create_dir_if_not_exists, log_range_accumulated_errors


def generate_multi_plot(model, model_params, epoch, starting_point, nr_prediction_steps, all_reset_indices,
                        nr_sequences_to_plot, nr_rows, source="TRAIN"):
    if source == "TRAIN":
        examples_batch, seq_addresses = model_params.dataset.get_train_plot_examples(nr_sequences_to_plot)
    elif source == "VAL":
        examples_batch, seq_addresses = model_params.dataset.get_val_plot_examples(nr_sequences_to_plot)

    all_prediction_losses_norm_mean = {source: []}

    dir_name_root = "{}/{}".format(model_params.model_path, seq_addresses[0].source)

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

        for i in range(nr_sequences_to_plot):
            image_name = "{}_E:{}".format(str(seq_addresses[i]), epoch)

            prediction_errors, predicted_sequences, vlines_coords = get_sequence_prediction(model, model_params,
                                                                                            examples_batch[i],
                                                                                            nr_prediction_steps,
                                                                                            starting_point,
                                                                                            reset_indices)

            prediction_losses_normalized += get_normalized_pred_errors(prediction_errors, reset_indices_)
            prediction_losses.append(prediction_errors)
            sequence_predictions.append(predicted_sequences)
            sequence_names.append(image_name)
            original_sequences.append(examples_batch[i])
            vlines_coords_list.append(vlines_coords)

        generate_subplots(original_sequences, sequence_predictions, vlines_coords_list, sequence_names,
                          starting_point + model_params.slice_length, plt_path,
                          model_params.gamma_windows_in_trial, nr_rows)
        generate_errors_subplots(prediction_losses, vlines_coords_list, sequence_names,
                                 starting_point + model_params.slice_length, plt_path,
                                 model_params.gamma_windows_in_trial, nr_rows)

        prediction_losses_normalized_seq_avg = prediction_losses_normalized / nr_sequences_to_plot
        log_range_accumulated_errors(epoch, prediction_losses_normalized_seq_avg, reset_indices_, dir_name)
        all_prediction_losses_norm_mean[source].append(np.mean(prediction_losses_normalized_seq_avg))

    return all_prediction_losses_norm_mean


def get_reset_indices_array(model_params, reset_indices):
    reset_indices_np_sorted = (np.array(list(reset_indices)) - model_params.slice_length)
    reset_indices_np_sorted.sort()
    pos_reset_indices = reset_indices_np_sorted > 0
    reset_indices_ = reset_indices_np_sorted[pos_reset_indices]
    return reset_indices_, reset_indices_np_sorted


def generate_subplots(original_sequences, sequence_predictions, vlines_coords_list, sequence_names,
                      prediction_starting_point, save_path, gamma_windows, nr_rows):
    nr_cols = len(sequence_predictions) // nr_rows
    prediction_length = sequence_predictions[0].size
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

        subplot.plot(predictions_x_indices, sequence_predictions[i], label="Predicted sequence", color="red",
                     linewidth=1)

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
        subplot.plot(prediction_losses[i], linewidth=1, label="Prediction accumulated errors", color="blue")
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


def show_and_plot(plot_save_path, plot_title, show):
    plt.title(plot_title)
    plt.savefig(plot_save_path)
    plt.show() if show else None
    plt.close()


def plot_samples(generated_images, save_path, name, epoch):
    nr_cols = np.int(np.sqrt(generated_images.shape[0]))
    nr_rows = generated_images.shape[0] // nr_cols

    fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(20, 20), num=name)
    for i, subplot in enumerate(subplots.flatten()):
        subplot.imshow(generated_images[i, :, :], cmap="gray")
        subplot.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("{}/{}-Epoch:{}.png".format(save_path, name, epoch), format="png")
    plt.close()
