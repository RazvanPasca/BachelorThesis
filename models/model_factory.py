from keras import Model, Input, optimizers, losses, metrics
from keras.layers import Dense, Flatten, K
from keras.losses import mse

import TrainingConfiguration
from datasets.datasets_utils import ModelType
from models.deconv_decoder import deconv_decoder
from models.wavenet_encoder import get_wavenet_encoder
from models.z_layer import get_z_layer


def get_simple_model_loss(model_args: TrainingConfiguration):
    if model_args.loss == "MSE":
        model_loss = losses.MSE
    elif model_args.loss == "MAE":
        model_loss = losses.MAE
    elif model_args.loss == "CE":
        model_loss = losses.sparse_categorical_crossentropy
    else:
        raise ValueError("Give a proper loss function")
    return model_loss


def reconstruction_loss(y_true, y_pred):
    rec_loss = mse(K.flatten(y_true), K.flatten(y_pred))
    return rec_loss


def get_vae_kl_loss(y_true, y_pred):
    kl_loss = K.mean(- 0.5 * K.sum(1 + y_pred - K.square(y_true) - K.exp(y_pred), axis=-1))
    return kl_loss


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
        output, layers = deconv_decoder(model_args, z_layer)

        if model_args.use_vae:
            decoder_input = Input(shape=(model_args.z_dim,))

            generator_output = layers[0](decoder_input)
            for layer in layers[1:]:
                generator_output = layer(generator_output)

            generator = Model(decoder_input, generator_output)
            setattr(model_args, "generator", generator)

    elif model_args.model_type == ModelType.BRIGHTNESS or model_args.model_type == ModelType.EDGES:
        z_layer= get_z_layer(model_args, encoder_output)
        output = Dense(1, name="Regression")(z_layer)

    elif model_args.model_type == ModelType.NEXT_TIMESTEP:
        net = Flatten()(encoder_output)
        model_metrics = [metrics.sparse_categorical_accuracy]
        output = Dense(model_args.dataset.nr_bins, activation='softmax', name="Softmax")(net)

    model = Model(inputs=input, outputs=output)

    if model_args.use_vae:
        loss = reconstruction_loss
        model_metrics = [reconstruction_loss]
    else:
        loss = get_simple_model_loss(model_args)

    optimizer = optimizers.adam(lr=model_args.lr, clipvalue=model_args.clip_value)
    model.compile(loss=loss, optimizer=optimizer, metrics=model_metrics)

    return model
