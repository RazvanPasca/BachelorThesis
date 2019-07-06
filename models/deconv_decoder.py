from keras.backend import tf
from keras.engine import Layer, InputSpec
from keras.layers import Dense, Reshape, Conv2D, LeakyReLU, UpSampling2D, Conv2DTranspose
from keras.regularizers import l2
from train.TrainingConfiguration import TrainingConfiguration


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


def deconv_decoder(model_args: TrainingConfiguration, net):
    seed_img_size = model_args.output_image_size // (2 ** (len(model_args.deconv_layers) - 1))

    layers = []

    layers.append(Dense(model_args.deconv_layers[0] * seed_img_size * seed_img_size, name="Dense_before_deconv"))
    layers.append(Reshape((seed_img_size, seed_img_size, model_args.deconv_layers[0])))
    layers.append(LeakyReLU())

    for nr_deconv_filters in model_args.deconv_layers[1:]:
        conv2d = Conv2DTranspose(nr_deconv_filters, (5, 5), padding='same', strides=2,
                                 kernel_regularizer=l2(model_args.regularization_coef))
        leaky = LeakyReLU()
        layers.append(conv2d)
        layers.append(leaky)

    layers.append(Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding='same',
                         kernel_regularizer=l2(model_args.regularization_coef),
                         name="Output_Conv"))

    decoder_output = layers[0](net)
    for layer in layers[1:]:
        decoder_output = layer(decoder_output)

    return decoder_output, layers


class DeconvDecoder(Layer):
    def __init__(self, model_args, **kwargs):
        self.model_args = model_args
        self.name = "Deconvolutional_Decoder"
        self.seed_img_size = model_args.output_image_size // (2 ** (len(model_args.deconv_layers) - 1))
        super(DeconvDecoder, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = Dense(self.model_args.deconv_layers[0] * self.seed_img_size * self.seed_img_size,
                           name="Dense_before_deconv")
        # self.add_weight(self.dense.get_weights())

        self.reshape = Reshape((self.seed_img_size, self.seed_img_size, self.model_args.deconv_layers[0]))
        # self.add_weight(self.reshape.get_weights())

        self.leaky_1 = LeakyReLU()
        # self.add_weight(self.leaky_1.get_weights())

        self.deconv_layers = [(Conv2DTranspose(nr_deconv_filters, (5, 5), padding='same', strides=2,
                                               kernel_regularizer=l2(self.model_args.regularization_coef)),
                               LeakyReLU())
                              for nr_deconv_filters in self.model_args.deconv_layers[1:]]

        # for deconv,leaky in self.deconv_layers:
        # self.add_weight(deconv.get_weights())
        # self.add_weight(leaky.get_weights())

        self.output_layer = Conv2D(filters=1, kernel_size=(5, 5), strides=1, padding='same',
                                   kernel_regularizer=l2(self.model_args.regularization_coef),
                                   name="Output_Conv")

        # self.add_weight(self.output_layer.get_weights())

        assert len(input_shape) == 2
        super(DeconvDecoder, self).build(input_shape)

    def call(self, x):
        generator = self.dense(x)
        generator = self.reshape(generator)

        generator = self.leaky_1(generator)

        for deconv_layer, activation in self.deconv_layers:
            generator = deconv_layer(generator)
            generator = activation(generator)

        output = self.output_layer(generator)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.model_args.output_image_size, self.model_args.output_image_size, 1)
