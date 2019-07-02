from keras import losses, Input, Model, optimizers, metrics
from keras.backend import tf
from keras.engine import Layer, InputSpec
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense, Reshape, Lambda, BatchNormalization, \
    Conv2D, LeakyReLU, UpSampling2D
from keras.regularizers import l2



def wavenet_block(n_filters, filter_size, dilation_rate, regularization_coef, first=False):
    def f(input_):
        if first:
            residual = Lambda(lambda x: x[:, :, 0], output_shape=(1,))(input_)
            residual = Reshape(target_shape=(-1, 1))(residual)
        else:
            residual = input_
        tanh_out = Conv1D(filters=n_filters,
                          kernel_size=filter_size,
                          dilation_rate=dilation_rate,
                          padding='same',
                          activation='tanh',
                          kernel_regularizer=l2(regularization_coef),
                          name="Tanh_{}".format(dilation_rate))(input_)

        sigmoid_out = Conv1D(filters=n_filters,
                             kernel_size=filter_size,
                             dilation_rate=dilation_rate,
                             padding='same',
                             activation='sigmoid',
                             kernel_regularizer=l2(regularization_coef),
                             name="Sigmoid_{}".format(dilation_rate))(input_)

        merged = Multiply(name="Gate_{}".format(dilation_rate))([tanh_out, sigmoid_out])

        skip_out = Conv1D(filters=n_filters * 2,
                          kernel_size=1,
                          padding='same',
                          kernel_regularizer=l2(regularization_coef),
                          name="Skip_Conv_{}".format(dilation_rate))(merged)

        out = Conv1D(filters=n_filters,
                     kernel_size=1,
                     padding='same',
                     kernel_regularizer=l2(regularization_coef),
                     name="Res_Conv_{}".format(dilation_rate))(merged)

        full_out = Add(name="Block_{}".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return f


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


def get_full_model(nr_filters, input_shape, nr_layers, lr, loss, clipvalue, skip_conn_filters,
                   regularization_coef, z_dim, output_type, img_size=64, generator_filter_size=64,
                   nr_classes=3):
    input_ = Input(shape=input_shape)
    net = get_wavenet_encoder(input_, nr_filters, nr_layers, regularization_coef, skip_conn_filters)

    net = Flatten()(net)
    net = LeakyReLU()(net)
    net = Dense(z_dim)(net)
    net = LeakyReLU()(net)

    model = get_model_output_stage(clipvalue, generator_filter_size, img_size, input_, loss, lr, output_type, net,
                                   nr_classes,
                                   regularization_coef)
    return model


def get_wavenet_encoder(input_, nr_filters, nr_layers, l2_coef, skip_conn_filters):
    A, B = wavenet_block(nr_filters, 3, 1, regularization_coef=l2_coef, first=True)(input_)
    skip_connections = [B]
    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 3, dilation_rate, regularization_coef=l2_coef)(A)
        skip_connections.append(B)
    net = Add(name="Skip_Merger")(skip_connections)
    net = LeakyReLU()(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(l2_coef), name="Skip_FConv_1")(net)
    net = LeakyReLU()(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(l2_coef), name="Skip_FConv_2")(net)
    return net



