from keras.engine import Layer, InputSpec
from keras.layers import Activation, BatchNormalization, \
    Conv2D, UpSampling2D
from keras.regularizers import l2
from keras.backend import tf


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


def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(1, 1), regularization_coef=0.0, bn_relu=True):
    """Layers used during upsampling"""
    padding = kernel_size[0] // 2
    u = UpSampling2D((2, 2), interpolation="nearest", data_format="channels_last")(layer_input)
    u = ReflectionPadding2D((padding, padding))(u)
    u = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid',
               kernel_regularizer=l2(regularization_coef))(u)
    if bn_relu:
        u = BatchNormalization(momentum=0.9)(u)
        u = Activation('relu')(u)
    return u


def get_deconv_decoder(input, model_args):
    pass
