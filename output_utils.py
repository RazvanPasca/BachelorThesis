import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def decode_model_output(model_logits, model_params):
    if model_params.get_classifying() == 2:
        return get_bin_output(model_logits[1], model_params), model_logits[0][0]
    elif model_params.get_classifying() == 1:
        return get_bin_output(model_logits, model_params),
    else:
        return model_logits,


def get_bin_output(model_logits, model_params):
    bin_index = np.argmax(model_logits)
    return (model_params.dataset.bins[bin_index - 1] + model_params.dataset.bins[bin_index]) / 2


def get_normalized_erros_per_sequence(losses, reset_indices):
    """
    :param losses: a list with the accumulated losses computed for model generation, the loss and generation process
                    being reset at every position found in reset_indices
    :param reset_indices: a list with the indices where the generation process is reset
    :return: the normalized values right before the reset_indices normalized by the number of predictions made up
            to that point from the previous reset position
            Will contain len(reset_indices) values
    """

    values_np = losses[reset_indices - 1]
    reset_indices_windows = np.ediff1d(reset_indices, to_begin=reset_indices[0])
    normalized_values = values_np / reset_indices_windows
    return normalized_values


"""The functions below are just for guidance. No longer used"""


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


def log_metrics_to_text(metrics, classes, fname):
    formatter = ",".join("{:25}" for _ in range(len(metrics) - 1)) + "\n"
    with open(fname, "w+") as f:
        keys = list(metrics.keys())
        f.write(formatter.format("Class", *keys[:-2]), )
        for i in range(len(classes)):
            f.write(formatter.format(classes[i], *[metrics[key][i] for key in keys[:-2]]))
        f.write(str([(key, val) for key, val in list(metrics.items())[-2:]]))


def compute_conf_matrix(model, X, Y):
    pred = model.predict(X)
    max_pred = np.argmax(pred, axis=1)
    max_y = Y
    cnf_mat = confusion_matrix(max_y, max_pred)

    predicted_positives = np.sum(cnf_mat, axis=0)
    actual_positives = np.sum(cnf_mat, axis=1)
    cnf_mat_diag = np.diag(cnf_mat)
    precision = np.around(cnf_mat_diag / predicted_positives, decimals=4)
    recall = np.around(cnf_mat_diag / actual_positives, decimals=4)
    f1 = np.around(2 * (precision * recall) / (precision + recall + 0.00001), decimals=4)
    diag_sum = np.sum(cnf_mat_diag)
    micro_precision = np.around(diag_sum / np.sum(predicted_positives), decimals=4)
    micro_recall = np.around(diag_sum / np.sum(actual_positives), decimals=4)
    metrics = {"precision": precision, "recall": recall, "f1": f1, "micro_precision": micro_precision,
               "micro_recall": micro_recall}
    return metrics, cnf_mat


