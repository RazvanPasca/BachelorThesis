import datetime

from keras.models import load_model

from plot_utils import get_partial_generated_sequences, generate_prediction_name, \
    create_dir_if_not_exists
from train_model import configure_gpu
from training_parameters import ModelTrainingParameters


def test_model(model_params):
    model_path = model_params.model_path

    model = load_model(model_path + '.h5')
    model.summary()
    pred_seqs = model_params.dataset.prediction_sequences

    name_prefix = "Generated:" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H:%M")

    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            dir_name = "/" + name_prefix + "/" + addr["SOURCE"] + "/"
            create_dir_if_not_exists(model_path + dir_name)
            image_name = dir_name + image_name
            for nr_generated_steps in range(1, 20, 2):
                get_partial_generated_sequences(model, model_params, sequence, nr_predictions=50,
                                                image_name=image_name,
                                                starting_point=1200,
                                                nr_generated_steps=nr_generated_steps)

    print("Finished testing model")


if __name__ == '__main__':
    configure_gpu(0)
    model_path = "/home/pasca/School/Licenta/Naturix/LFP_models/Wavenet_L:7_Ep:50_StpEp:1850_Lr:1e-05_BS:32_Fltrs:16_SkipFltrs:32_L2:0.0001_FS:8_CAT_Clip:True_Rnd:True/2019-03-13 16:30"
    model_parameters = ModelTrainingParameters(model_path)
    test_model(model_parameters)
