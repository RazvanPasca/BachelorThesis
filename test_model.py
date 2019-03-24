import datetime
import os

import numpy as np
from keras.models import load_model

from plot_utils import get_predictions_with_losses, generate_prediction_name, \
    create_dir_if_not_exists
from tf_utils import configure_gpu
from training_parameters import ModelTrainingParameters


def get_error_estimates(model, model_parameters, nr_of_estimates, generated_window_size, file_to_save):
    sequence, addr = model_parameters.dataset.get_random_sequence_from("VAL")

    avg_prediction_losses = np.zeros(nr_of_estimates)
    for estimate in range(nr_of_estimates):
        starting_point = np.random.choice(sequence.size - model_parameters.frame_size - generated_window_size)
        prediction_losses = get_predictions_with_losses(model, model_parameters, sequence, generated_window_size,
                                                        image_name="bla",
                                                        starting_point=starting_point,
                                                        generated_window_size=generated_window_size,
                                                        plot=False)
        avg_prediction_losses[estimate] = prediction_losses[-1]

    avg_prediction_losses = np.mean(avg_prediction_losses)
    normalized_prediction_losses = avg_prediction_losses / generated_window_size

    with open(file_to_save, "a") as f:
        f.write(
            "{:>10}, {:>10}, {:12.4}\n".format(nr_of_estimates, generated_window_size, normalized_prediction_losses))


def test_model(model_params):
    model_path = model_params.model_path

    model = load_model(model_path + '.h5')
    model.summary()
    pred_seqs = model_params.dataset.prediction_sequences

    name_prefix = "Generated:" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H:%M")

    file_path = model_params.model_path + "/Error_statistics.txt"
    prepare_file_for_writing(file_path, "Nr estimates, Generated window size, Normalized average error\n")

    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            dir_name = name_prefix + "/" + addr["SOURCE"] + "/"
            create_dir_if_not_exists(os.path.join(model_path, dir_name))
            image_name = os.path.join(dir_name, image_name)
            for generated_window_size in range(1, 100, 5):
                get_predictions_with_losses(model, model_params, sequence, nr_predictions=100,
                                            image_name=image_name, starting_point=1200 - model_params.frame_size,
                                            generated_window_size=generated_window_size)

    for generated_window_size in range(1, 100, 5):
        get_error_estimates(model, model_params, 10, generated_window_size,
                            file_path)

    print("Finished testing model")


def prepare_file_for_writing(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


if __name__ == '__main__':
    configure_gpu(0)
    model_path = "/home/gabir/Repos/BachelorThesis/MouseControl/Channels:[1]/WvNet_L:7_Ep:300_StpEp:1603.0_Lr:1e-05_BS:32_Fltrs:32_SkipFltrs:64_L2:0.0001_Norm:Zsc_CAT:512_Clip:True_Rnd:True/2019-03-23 16:08"
    model_parameters = ModelTrainingParameters(model_path)
    test_model(model_parameters)
