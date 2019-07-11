import os

from keras.engine.saving import load_model

from datasets.datasets_utils import ModelType
from model_test_utils.test_ae_model import test_ae_model
from model_test_utils.test_regression_models import test_regression_models
from model_test_utils.test_vae_model import test_vae_model, test_vae_decoder
from models.model_factory import reconstruction_loss
from models.z_layer import KLDivergenceLayer
from train.TrainingConfiguration import TrainingConfiguration


def load_model_from_folder(params, model_folder, model_name):
    custom_objects = None
    if params.use_vae:
        custom_objects = {"KLDivergenceLayer": KLDivergenceLayer, "reconstruction_loss": reconstruction_loss}
    model = load_model(os.path.join(model_folder, model_name), custom_objects)
    return model


def load_params_from_folder(folder):
    config_path = os.path.join(folder, "config_file.py")
    model_parameters = TrainingConfiguration(config_path)
    return model_parameters


def test_model(model_folder):
    params = load_params_from_folder(model_folder)

    if params.model_type == ModelType.BRIGHTNESS or params.model_type == ModelType.EDGES:
        model = load_model_from_folder(params, model_folder, "best_model.h5")
        test_regression_models(model, nr_samples, params)

    if params.model_type == ModelType.IMAGE_REC:
        model = load_model_from_folder(params, model_folder, "best_model.h5")
        if params.use_vae:
            test_vae_model(model, params)
            model = load_model_from_folder(params, model_folder, "decoder.h5", )
            test_vae_decoder(model, params)
        else:
            test_ae_model(model, params)


if __name__ == "__main__":
    nr_samples = 3000
    MY_MODEL_PATH = "/home/pasca/School/Licenta/Naturix/Results_after_refactor/ModelType.IMAGE_REC-VAE/Movies:[0]/SplitStrategy.TRIALS-SlicingStrategy.CONSECUTIVE-WinL:100-Stacked:True/EncL:7_Dil:False_Ep:400_StpEp:84.0_Perc:0.003_Lr:1e-05_BS:32_Fltrs:16_SkipFltrs:16_ZDim:10_L2:0.001_Loss:MAE_GradClip:None_LPass:None_DecL:[64, 32, 16, 8, 4]_Kl:0.001_RelDif:False/Pid:21854__2019-07-11 15:32_Seed:42"

    test_model(MY_MODEL_PATH)
