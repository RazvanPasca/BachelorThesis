import datetime

import numpy as np
from keras.models import load_model

from plot_utils import get_sequence_prediction, generate_prediction_name, \
    create_dir_if_not_exists, prepare_file_for_writing
from output_utils import get_normalized_prediction_losses
from tf_utils import configure_gpu
from training_parameters import ModelTrainingParameters


def get_error_estimates(model, model_parameters, nr_of_estimates, generated_window_size, file_to_save, source):
    sequence, addr = model_parameters.dataset.get_random_sequence_from(source)

    avg_prediction_losses = np.zeros(nr_of_estimates)
    for estimate in range(nr_of_estimates):
        starting_point = np.random.choice(sequence.size - model_parameters.frame_size - generated_window_size)
        prediction_losses, _ = get_sequence_prediction(model,
                                                       model_parameters,
                                                       sequence,
                                                       generated_window_size,
                                                       image_name="bla",
                                                       starting_point=starting_point,
                                                       generated_window_size=generated_window_size,
                                                       plot=False)
        avg_prediction_losses[estimate] = prediction_losses[-1]

    normalized_prediction_losses = get_normalized_prediction_losses(generated_window_size, avg_prediction_losses)

    with open(file_to_save, "a") as f:
        f.write(
            "{:>10}, {:>10}, {:12.4}\n".format(nr_of_estimates, generated_window_size, normalized_prediction_losses))


def test_model(model_params):
    model_path = model_params.model_path

    model = load_model(model_path + '/best_model.h5')
    model.summary()
    pred_seqs = model_params.dataset.prediction_sequences

    name_prefix = "Generated:" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H:%M")

    print("Started testing model...")
    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_prefix = generate_prediction_name(addr)
            image_prefix = name_prefix + "/" + addr["SOURCE"] + "/" + image_prefix
            image_name = image_prefix + "/"

            create_dir_if_not_exists(model_params.model_path + "/" + image_prefix)

            for generated_window_size in range(1, 1000, 50):
                get_sequence_prediction(model,
                                        model_params,
                                        sequence,
                                        nr_predictions=1000,
                                        image_name=image_name,
                                        starting_point=1500 - model_params.frame_size,
                                        generated_window_size=generated_window_size)

    for source in ["VAL", "TRAIN"]:
        file_path = model_params.model_path + "/Error_statistics_{}.txt".format(source)
        prepare_file_for_writing(file_path, "Nr estimates, Generated window size, Normalized average error\n")
        for generated_window_size in range(1, 100, 2):
            get_error_estimates(model,
                                model_params,
                                250,
                                generated_window_size,
                                file_path, source)

    print("Finished testing model")


if __name__ == '__main__':
    model_path = "/home/pasca/School/Licenta/Naturix/LFP_models/MouseControl/Movies:None/Channels:[32]/WvNet_L:6_Ep:10_StpEp:80.0_Lr:1e-05_BS:32_Fltrs:32_SkipFltrs:64_L2:0.0001_Norm:Zsc_CAT:256_Clip:0.5_Rnd:True_LPass:40/2019-04-08 14:51"
    model_parameters = ModelTrainingParameters(model_path)
    configure_gpu(model_parameters.gpu)
    test_model(model_parameters)
