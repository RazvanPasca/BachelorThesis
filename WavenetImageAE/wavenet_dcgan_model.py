from keras import losses, Input, Model, optimizers
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense, Reshape, Lambda, BatchNormalization, \
    Conv2DTranspose, LeakyReLU
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


def deconv2d(layer_input, filters=256, kernel_size=(5, 5), strides=(2, 2), regularization_coef=0.0, bn_relu=True):
    """Layers used during upsampling"""
    u = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides,
                        kernel_regularizer=l2(regularization_coef),
                        padding='same')(layer_input)
    if bn_relu:
        u = BatchNormalization(momentum=0.9)(u)
        u = Activation('relu')(u)
    return u


def get_wavenet_dcgan_model(nr_filters, input_shape, nr_layers, lr, loss, clipvalue, skip_conn_filters,
                            regularization_coef, z_dim, img_size=64, generator_filter_size=64):
    input_ = Input(shape=input_shape)
    A, B = wavenet_block(nr_filters, 3, 1, regularization_coef=regularization_coef, first=True)(input_)
    skip_connections = [B]

    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 3, dilation_rate, regularization_coef=regularization_coef)(A)
        skip_connections.append(B)

    net = Add(name="Skip_Merger")(skip_connections)
    net = LeakyReLU()(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(regularization_coef), name="Skip_FConv_1")(net)
    net = LeakyReLU()(net)
    net = Conv1D(skip_conn_filters, 1, kernel_regularizer=l2(regularization_coef), name="Skip_FConv_2")(net)
    net = Flatten()(net)
    net = LeakyReLU()(net)

    net = Dense(z_dim)(net)
    net = LeakyReLU()(net)

    seed_img_size = img_size // 16
    generator = Dense(16 * generator_filter_size * seed_img_size * seed_img_size)(net)
    generator = Reshape((seed_img_size, seed_img_size, generator_filter_size * 16))(generator)
    # generator = BatchNormalization()(generator)
    generator = LeakyReLU()(generator)
    generator = deconv2d(generator, filters=generator_filter_size * 8, regularization_coef=regularization_coef,
                         bn_relu=False)
    generator = deconv2d(generator, filters=generator_filter_size * 4, regularization_coef=regularization_coef,
                         bn_relu=False)
    generator = deconv2d(generator, filters=generator_filter_size * 2, regularization_coef=regularization_coef,
                         bn_relu=False)
    output = deconv2d(generator, filters=1, regularization_coef=regularization_coef, bn_relu=False)

    model = Model(inputs=input_, outputs=output)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=losses.MSE if loss.lower() == "mse" else losses.MAE, optimizer=optimizer)
    return model
