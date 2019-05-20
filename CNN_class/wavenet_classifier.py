from keras import Input, Model, optimizers, metrics
from keras.activations import softmax
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense
from keras.regularizers import l2


def wavenet_block(n_filters, filter_size, dilation_rate, regularization_coef):
    def f(input_):
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


def get_wavenet_model(nr_filters, input_shape, nr_layers, lr, clipvalue, skip_conn_filters,
                      regularization_coef, nr_output_classes):
    input_ = Input(shape=input_shape)
    A, B = wavenet_block(nr_filters, 2, 1, regularization_coef=regularization_coef, first=True)(input_)
    skip_connections = [B]

    for i in range(1, nr_layers):
        dilation_rate = 2 ** i
        A, B = wavenet_block(nr_filters, 2, dilation_rate, regularization_coef=regularization_coef)(A)
        skip_connections.append(B)

    net = Add(name="Skip_Merger")(skip_connections)
    net = Activation('relu')(net)
    net = Conv1D(skip_conn_filters, 1, activation='relu', padding="same",
                 kernel_regularizer=l2(regularization_coef), name="Skip_FConv_1")(net)
    net = Conv1D(skip_conn_filters, 1, padding="same",
                 kernel_regularizer=l2(regularization_coef), name="Skip_FConv_2")(net)
    net = Flatten()(net)

    output = Dense(nr_output_classes, activation=softmax, name="Sfmax")(net)

    model = Model(inputs=input_, outputs=output)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer,
                  metrics=[metrics.sparse_categorical_accuracy])
    return model
