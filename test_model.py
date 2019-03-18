import datetime

from keras.models import load_model

from plot_utils import get_partial_generated_sequences, generate_prediction_name, \
    create_dir_if_not_exists
from training_parameters import ModelTrainingParameters


def test_model(model_params, model_path=None):
    if model_path is None:
        model_path = model_params.model_path

    model = load_model(model_path + '.h5')
    model.summary()
    pred_seqs = model_params.dataset.prediction_sequences

    name_prefix = "/Generated:" + datetime.datetime.now().strftime(
        "%Y-%m-%d-%H:%M")
    create_dir_if_not_exists(model_parameters.model_path + name_prefix)

    for source in pred_seqs:
        for sequence, addr in pred_seqs[source]:
            image_name = generate_prediction_name(addr)
            image_name = name_prefix + "/" + "E:{}_".format("TrainEnd") + image_name
            for nr_generated_steps in range(1, 50, 2):
                get_partial_generated_sequences(model, model_params, sequence, nr_predictions=100,
                                                image_name=image_name,
                                                starting_point=1200 - model_params.frame_size - 10,
                                                nr_generated_steps=nr_generated_steps)

    print("Finished testing model")


if __name__ == '__main__':
    model_path = "/home/pasca/School/Licenta/Naturix/LFP_models/Wavenet_L:7_Ep:50_StpEp:1850_Lr:1e-05_BS:32_Fltrs:16_SkipFltrs:32_L2:0.0001_FS:8_CAT_Clip:True_Rnd:True/2019-03-13 16:30"
    model_parameters = ModelTrainingParameters(model_path)
    test_model(model_parameters, model_path)
