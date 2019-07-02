from keras.layers import Dense, Flatten, LeakyReLU, Lambda, K

import TrainingConfiguration


def get_z_layer(model_args: TrainingConfiguration, encoder_output):
    z_layer = Flatten()(encoder_output)

    if model_args.use_vae:
        z_mean = Dense(model_args.z_dim)(encoder_output)
        z_log_sigma = Dense(model_args.z_dim)(encoder_output)
        z_layer = Lambda(sampling, output_shape=(model_args.z_dim,))([z_mean, z_log_sigma])
    else:
        z_layer = LeakyReLU()(z_layer)
        z_layer = Dense(model_args.z_dim, name="Z_layer")(z_layer)
        z_layer = LeakyReLU()(z_layer)
    return z_layer


def sampling(args):
    z_mean, z_log_sigma = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), mean=0., stddev=1.0)
    return z_mean + K.exp(z_log_sigma) * epsilon
