import numpy as np


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


def get_normalized_prediction_losses(generated_window_size, prediction_losses):
    prediction_losses_avg = np.mean(prediction_losses)
    # prediction_losses_normalized = prediction_losses_avg / generated_window_size
    return prediction_losses_avg


def get_cumulated_error_mean_per_sequence(losses, reset_indices, range):
    values = 0
    counter = 0
    for i, val in enumerate(losses[:-1]):
        if val > losses[i + 1]:
            values += val
            counter += 1
    return values / counter if counter != 0 else losses[-1]
