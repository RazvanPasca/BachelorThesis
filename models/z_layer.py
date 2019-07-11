from keras.engine import Layer
from keras.layers import Dense, Flatten, LeakyReLU, Lambda, K, ReLU

from train import TrainingConfiguration


class KLDivergenceLayer(Layer):
    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, kl_weight, *args, **kwargs):
        self.kl_weight = kl_weight
        self.name = "KL_Loss"
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs

        kl_batch = -self.kl_weight * .5 * K.sum(1 + log_var -
                                                K.square(mu) -
                                                K.exp(log_var), axis=-1)

        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs

    def get_config(self):
        config = {'kl_weight': self.kl_weight}
        base_config = super(KLDivergenceLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def get_z_layer(model_args: TrainingConfiguration, encoder_output):
    z_layer = Flatten()(encoder_output)

    if model_args.use_vae:
        z_layer = ReLU()(z_layer)
        z_mean = Dense(model_args.z_dim, name="Z_mean")(z_layer)
        z_log_sigma = Dense(model_args.z_dim, name="Z_log_sigma")(z_layer)

        z_mean, z_log_sigma = KLDivergenceLayer(kl_weight=model_args.kl_weight)([z_mean, z_log_sigma])

        z_layer = Lambda(sampling, output_shape=(model_args.z_dim,), name="Z_Sampler")([z_mean, z_log_sigma])
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
