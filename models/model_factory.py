from datasets.datasets_utils import ModelType
from models.wavenet_encoder import get_wavenet_encoder


def get_model(model_type):
    if model_type == ModelType.SCENE_CLASSIFICATION:
        encoder = get_wavenet_encoder()
        
    elif model_type == ModelType.IMAGE_REC:
        encoder = get_wavenet_encoder()
        decoder = get_deconv_decoder()

    elif model_type == ModelType.BRIGHTNESS or model_type == ModelType.EDGES:
        encoder = get_wavenet_encoder()

    elif model_type == ModelType.NEXT_TIMESTEP:
        encoder = get_wavenet_encoder()

    elif model_type == ModelType.CONDITION_CLASSIFICATION:
        encoder = get_wavenet_encoder()