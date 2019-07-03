from keras import Model, Input, optimizers, losses, metrics
from keras.layers import Dense, Flatten

import TrainingConfiguration
from datasets.datasets_utils import ModelType
from models.deconv_decoder import get_deconv_decoder
from models.wavenet_encoder import get_wavenet_encoder
from models.z_layer import get_z_layer


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


def get_model(model_args: TrainingConfiguration):
    input = Input(shape=model_args.input_shape)
    encoder_output = get_wavenet_encoder(input, model_args)

    output = None
    model_metrics = None

    if model_args.model_type == ModelType.SCENE_CLASSIFICATION:
        z_layer = get_z_layer(model_args, encoder_output)
        output = Dense(model_args.nr_output_classes, activation='softmax', name="Sfmax")(z_layer)
        model_metrics = [metrics.sparse_categorical_accuracy]

    elif model_args.model_type == ModelType.CONDITION_CLASSIFICATION:
        net = Flatten()(encoder_output)
        output = Dense(model_args.nr_output_classes, activation="softmax", name="Softmax")(net)
        model_metrics = [metrics.sparse_categorical_accuracy]

    elif model_args.model_type == ModelType.IMAGE_REC:
        z_layer = get_z_layer(model_args, encoder_output)
        output = get_deconv_decoder(model_args, z_layer, model_args.output_image_size)

    elif model_args.model_type == ModelType.BRIGHTNESS or model_args.model_type == ModelType.EDGES:
        z_layer = get_z_layer(model_args, encoder_output)
        output = Dense(1, name="Regression")(z_layer)

    elif model_args.model_type == ModelType.NEXT_TIMESTEP:
        net = Flatten()(encoder_output)
        model_metrics = [metrics.sparse_categorical_accuracy]
        output = Dense(model_args.dataset.nr_bins, activation='softmax', name="Softmax")(net)

    loss = get_model_loss(model_args.loss)
    model = Model(inputs=input, outputs=output)
    optimizer = optimizers.adam(lr=model_args.lr, clipvalue=model_args.clip_value)
    model.compile(loss=loss, optimizer=optimizer, metrics=model_metrics)

    return model
