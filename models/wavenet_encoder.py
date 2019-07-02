from keras import Input
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense, Reshape, Lambda, BatchNormalization, \
    Conv2D, LeakyReLU, UpSampling2D
from keras.regularizers import l2


def create_wavenet_layer(n_filters,
                         filter_size,
                         padding,
                         dilation_rate,
                         regularization_coef,
                         first=False):
    def layer(input_):
        if first:
            residual = Lambda(lambda x: x[:, :, 0], output_shape=(1,))(input_)
            residual = Reshape(target_shape=(-1, 1))(residual)
        else:
            residual = input_

        tanh_out = Conv1D(filters=n_filters,
                          kernel_size=filter_size,
                          dilation_rate=dilation_rate,
                          padding=padding,
                          activation='tanh',
                          kernel_regularizer=l2(regularization_coef),
                          name="Tanh_{}".format(dilation_rate))(input_)

        sigmoid_out = Conv1D(filters=n_filters,
                             kernel_size=filter_size,
                             dilation_rate=dilation_rate,
                             padding=padding,
                             activation='sigmoid',
                             kernel_regularizer=l2(regularization_coef),
                             name="Sigmoid_{}".format(dilation_rate))(input_)

        merged = Multiply(name="Gate_{}".format(dilation_rate))([tanh_out, sigmoid_out])

        skip_out = Conv1D(filters=n_filters * 2,
                          kernel_size=1,
                          padding=padding,
                          kernel_regularizer=l2(regularization_coef),
                          name="Skip_Conv_{}".format(dilation_rate))(merged)

        out = Conv1D(filters=n_filters,
                     kernel_size=1,
                     padding=padding,
                     kernel_regularizer=l2(regularization_coef),
                     name="Res_Conv_{}".format(dilation_rate))(merged)

        full_out = Add(name="Block_{}".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return layer


def get_wavenet_encoder(
        nr_filters,
        input_shape,
        nr_layers,
        padding,
        skip_conn_filters,
        regularization_coef):
    input = Input(shape=input_shape)
    wavenet_layer = create_wavenet_layer(
        n_filters=nr_filters,
        filter_size=2,
        dilation_rate=1,
        padding=padding,
        regularization_coef=regularization_coef,
        first=True)
    prev_layer_output, prev_layer_skip_out = wavenet_layer(input)
    skip_connections = [prev_layer_skip_out]

    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        wavenet_layer = create_wavenet_layer(
            n_filters=nr_filters,
            filter_size=2,
            dilation_rate=dilation_rate,
            padding=padding,
            regularization_coef=regularization_coef)
        prev_layer_output, prev_layer_skip_out = wavenet_layer(prev_layer_output)
        skip_connections.append(prev_layer_skip_out)

    net = Add(name="Skip_Merger")(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(
        filters=skip_conn_filters,
        kernel_size=1,
        activation='relu',
        kernel_regularizer=l2(regularization_coef),
        name="Skip_FConv_1")(net)
    net = Conv1D(
        filters=skip_conn_filters,
        kernel_size=1,
        kernel_regularizer=l2(regularization_coef),
        name="Skip_FConv_2")(net)

    return net