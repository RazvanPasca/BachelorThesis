from keras.backend import tf
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Reshape, Conv2D, LeakyReLU, UpSampling2D
from keras.regularizers import l2


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


class DeconvDecoder(Layer):
    def __init__(self, model_args, **kwargs):
        self.model_args = model_args
        self.seed_img_size = model_args.output_image_size // (2 ** (len(model_args.deconv_layers) - 1))
        super(DeconvDecoder, self).__init__(**kwargs)

    def call(self, x):
        generator = Dense(self.model_args.deconv_layers[0] * self.seed_img_size * self.seed_img_size)(x)
        generator = Reshape((self.seed_img_size, self.seed_img_size, self.model_args.deconv_layers[0]))(generator)

        generator = LeakyReLU()(generator)

        for deconv_layer_filters in self.model_args.deconv_layers[1:]:
            generator = deconv2d(generator,
                                 filters=deconv_layer_filters,
                                 regularization_coef=self.model_args.regularization_coef)
            generator = LeakyReLU()(generator)

        output = Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding='same',
                        kernel_regularizer=l2(self.model_args.regularization_coef),
                        name="Output_Conv")(generator)

        return output
