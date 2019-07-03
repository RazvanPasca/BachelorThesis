import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


def decode_model_output(model_logits, model_params):
    return get_bin_output(model_logits, model_params)


def get_bin_output(model_logits, model_params):
    bin_index = np.argmax(model_logits)
    return (model_params.dataset.bins[bin_index - 1] + model_params.dataset.bins[bin_index]) / 2


def get_normalized_pred_errors(prediction_errors, reset_indices):
    """
    :param prediction_errors: a list with the accumulated losses computed for model generation, the loss and generation process
                    being reset at every position found in reset_indices
    :param reset_indices: a list with the indices where the generation process is reset
    :return: the normalized values right before the reset_indices normalized by the number of predictions made up
            to that point from the previous reset position
            Will contain len(reset_indices) values
    """

    values_np = prediction_errors[reset_indices - 1]
    reset_indices_windows = np.ediff1d(reset_indices, to_begin=reset_indices[0])
    normalized_values = values_np / reset_indices_windows
    return normalized_values


def compute_conf_matrix(model, X, Y_true):
    pred = model.predict(X)
    Y_predicted = np.argmax(pred, axis=1)
    cnf_mat = confusion_matrix(Y_true, Y_predicted)

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


def compute_confusion_matrix(svc, x, y):
    y_pred = svc.predict(x)
    conf_mat = confusion_matrix(y, y_pred)
    clas_rep = classification_report(y, y_pred)
    print(conf_mat)
    print(clas_rep)
    return conf_mat, clas_rep


def get_sequence_prediction(model, model_params, original_sequence, nr_predictions,
                            starting_point,
                            reset_indices):

    """
    Generates prediction given an original sequence as input

     Args:
            model: instance of the keras model used for predicting
            model_params: instance of the model_params associated with this training/testing session
            original_sequence: the original sequence from the dataset used as "seed" for the model
            nr_predictions: how many values we will predict in total
            starting_point: the place from which we start seeding the model with model_params.slice_length values
            reset_indices: when we reset the teacher forcing, starting with another original sequence seed

    Returns:
            prediction_losses: the accumulated errors associated with it
            predicted_sequences: the predicted sequence
            vlines_coords: a vlines object indicating the reset indices
    """

    predicted_sequences = np.zeros(nr_predictions)
    predictions_errors = np.zeros(nr_predictions)
    vlines_coords = []
    input_sequence = []

    for prediction_nr in range(nr_predictions):
        current_pos = starting_point + prediction_nr

        if current_pos + model_params.slice_length in reset_indices or prediction_nr == 0:
            seed_sequence = original_sequence[current_pos:current_pos + model_params.slice_length]
            input_sequence = np.reshape(seed_sequence, (-1, model_params.slice_length, 1))
            vlines_coords.append(current_pos + model_params.slice_length - 1)

        predicted_value = decode_model_output(model.predict(input_sequence), model_params)

        predicted_sequences[prediction_nr] = predicted_value
        # predicted_val_w_label = np.array(
        #     [predicted_value, original_sequence[current_pos + model_params.slice_length]])

        input_sequence = np.append(input_sequence[:, 1:, :], np.reshape(predicted_value, (-1, 1, 1)), axis=1)
        actual_value = original_sequence[current_pos + model_params.slice_length]

        curr_loss = np.abs(actual_value - predicted_value)
        predictions_errors[prediction_nr] = curr_loss if current_pos + model_params.slice_length in reset_indices \
            else predictions_errors[prediction_nr - 1] + curr_loss

    return predictions_errors, predicted_sequences, vlines_coords
