from keras.backend import tf
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Reshape, Conv2D, LeakyReLU, UpSampling2D
from keras.regularizers import l2

import TrainingConfiguration


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(1, 1), regularization_coef=0.0):
    """Layers used during upsampling"""
    padding = kernel_size[0] // 2
    u = UpSampling2D((2, 2), interpolation="nearest", data_format="channels_last")(layer_input)
    u = ReflectionPadding2D((padding, padding))(u)
    u = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid',
               kernel_regularizer=l2(regularization_coef))(u)
    return u


def get_deconv_decoder(model_args: TrainingConfiguration, net, output_image_size):
    seed_img_size = output_image_size // (2 ** (len(model_args.deconv_layers) - 1))

    generator = Dense(model_args.deconv_layers[0] * seed_img_size * seed_img_size)(net)
    generator = Reshape((seed_img_size, seed_img_size, model_args.deconv_layers[0]))(generator)

    generator = LeakyReLU()(generator)

    for deconv_layer_filters in model_args.deconv_layers[1:]:
        generator = deconv2d(generator,
                             filters=deconv_layer_filters,
                             regularization_coef=model_args.regularization_coef)
        generator = LeakyReLU()(generator)

    output = Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding='same',
                    kernel_regularizer=l2(model_args.regularization_coef),
                    name="Output_Conv")(generator)

    return output
