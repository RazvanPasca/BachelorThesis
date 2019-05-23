from keras import losses, Input, Model, optimizers
from keras.activations import softmax
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense, K, Reshape, Lambda, BatchNormalization, \
    UpSampling2D, Conv2D, Conv2DTranspose
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
                          padding='causal',
                          activation='tanh',
                          kernel_regularizer=l2(regularization_coef),
                          name="Tanh_{}".format(dilation_rate))(input_)

        sigmoid_out = Conv1D(filters=n_filters,
                             kernel_size=filter_size,
                             dilation_rate=dilation_rate,
                             padding='causal',
                             activation='sigmoid',
                             kernel_regularizer=l2(regularization_coef),
                             name="Sigmoid_{}".format(dilation_rate))(input_)

        merged = Multiply(name="Gate_{}".format(dilation_rate))([tanh_out, sigmoid_out])

        skip_out = Conv1D(filters=n_filters * 2,
                          kernel_size=1,
                          padding='causal',
                          kernel_regularizer=l2(regularization_coef),
                          name="Skip_Conv_{}".format(dilation_rate))(merged)

        out = Conv1D(filters=n_filters,
                     kernel_size=1,
                     padding='causal',
                     kernel_regularizer=l2(regularization_coef),
                     name="Res_Conv_{}".format(dilation_rate))(merged)

        full_out = Add(name="Block_{}".format(dilation_rate))([out, residual])
        return full_out, skip_out

    return f


def get_wavenet_dcgan_model(nr_filters, input_shape, nr_layers, lr, loss, clipvalue, skip_conn_filters,
                      regularization_coef, nr_output_classes, multiloss_weights=None, img_size = 64, generator_filter_size=64):

    input_ = Input(shape=input_shape)
    A, B = wavenet_block(nr_filters, 2, 1, regularization_coef=regularization_coef, first=True)(input_)
    skip_connections = [B]

    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 2, dilation_rate, regularization_coef=regularization_coef)(A)
        skip_connections.append(B)

    net = Add(name="Skip_Merger")(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(skip_conn_filters, 1, activation='relu', kernel_regularizer=l2(regularization_coef),
                 name="Skip_FConv_1")(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(regularization_coef), name="Skip_FConv_2")(net)
    net = Flatten()(net)

    def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), bn_relu=True):
        """Layers used during upsampling"""
        # u = UpSampling2D(size=2)(layer_input)
        u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding='same')(layer_input)
        if bn_relu:
            u = BatchNormalization(momentum=0.9)(u)
            u = Activation('relu')(u)
        return u

    generator = Dense(16 * generator_filter_size * img_size // 16 * img_size // 16, activation="relu")(net)
    generator = Reshape((img_size // 16, img_size // 16,  generator_filter_size * 16))(generator)
    # generator = BatchNormalization()(generator)
    generator = Activation('relu')(generator)
    generator = deconv2d(generator, filters=generator_filter_size * 8, bn_relu=False)
    generator = deconv2d(generator, filters=generator_filter_size * 4, bn_relu=False)
    generator = deconv2d(generator, filters=generator_filter_size * 2, bn_relu=False)
    output = deconv2d(generator, filters=1, bn_relu=False)
    # output = deconv2d(generator, filters=1, kernel_size=(3, 3), strides=(1, 1), bn_relu=False)

    # output = Activation('tanh')(generator)

    model = Model(inputs=input_, outputs=output)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=losses.MSE, optimizer=optimizer)
    return model
