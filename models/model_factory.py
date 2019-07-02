from keras import Model, Input, optimizers, metrics, losses
from keras.layers import Flatten, LeakyReLU, Dense

from datasets.datasets_utils import ModelType
from models import ModelArguments
from models.deconv_decoder import get_deconv_decoder
from models.wavenet_encoder import get_wavenet_encoder


def get_model_loss(loss):
    if loss == "MSE":
        model_loss = losses.MSE
    elif loss == "MAE":
        model_loss = losses.MAE
    elif loss == "CE":
        model_loss = losses.sparse_categorical_crossentropy
    else:
        raise ValueError("Give a proper loss function")
    return model_loss


def get_model(model_args: ModelArguments):
    input = Input(shape=model_args.input_shape)
    encoder_output = get_wavenet_encoder(input, model_args)

    output = None
    mets = None

    if model_args.model_type == ModelType.SCENE_CLASSIFICATION:
        net = Flatten()(encoder_output)
        net = LeakyReLU()(net)
        net = Dense(model_args.z_dim, name="Final_dense")(net)
        net = LeakyReLU()(net)

        output = Dense(model_args.nr_output_classes, activation='softmax', name="Sfmax")(net)
        mets = [metrics.sparse_categorical_accuracy]

    elif model_args.model_type == ModelType.IMAGE_REC:
        output = get_deconv_decoder(encoder_output, model_args)

    elif model_args.model_type == ModelType.BRIGHTNESS or \
            model_args.model_type == ModelType.EDGES:
        output = Dense(1, name="Regression")(encoder_output)

    elif model_args.model_type == ModelType.NEXT_TIMESTEP:
        encoder_output = get_wavenet_encoder(input, model_args)
        net = Flatten()(encoder_output)

        output = Dense(model_args.nr_output_classes, activation='softmax', name="Softmax")(net)

    elif model_args.model_type == ModelType.CONDITION_CLASSIFICATION:
        output = Dense(model_args.nr_output_classes, activation="softmax", name="Softmax")(encoder_output)

    loss = get_model_loss(model_args.loss)
    model = Model(inputs=input, outputs=output)
    optimizer = optimizers.adam(lr=model_args.lr, clipvalue=model_args.clip_value)
    model.compile(loss=loss, optimizer=optimizer, metrics=mets)
