from keras import losses, Input, Model, optimizers
from keras.activations import softmax
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense
from keras.regularizers import l2


def wavenet_block(n_filters, filter_size, dilation_rate, regularization_coef):
    def f(input_):
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


def get_wavenet_model(nr_filters, input_size, nr_layers, lr, loss, clipvalue, skip_conn_filters,
                      regularization_coef, nr_output_classes, multiloss_weights=None):
    if loss == "MSE":
        model_loss = losses.MSE
    elif loss == "MAE":
        model_loss = losses.MAE
    elif loss == "CE":
        model_loss = losses.sparse_categorical_crossentropy
    elif loss == "MSE_CE":
        model_loss = {"Regression": "mean_squared_error", "Sfmax": "sparse_categorical_crossentropy"}
        assert multiloss_weights is not None

    elif loss == "MAE_CE":
        model_loss = {"Regression": "mean_absolute_error", "Sfmax": "sparse_categorical_crossentropy"}
        assert multiloss_weights is not None

    else:
        raise ValueError("Give a proper loss function")

    input_ = Input(shape=(input_size, 1))
    A, B = wavenet_block(nr_filters, 2, 1, regularization_coef=regularization_coef)(input_)
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

    outputs = []

    if model_loss is losses.sparse_categorical_crossentropy:
        outputs.append(Dense(nr_output_classes, activation=softmax, name="Sfmax")(net))
    else:
        regr_output = Dense(1, name="Regression", kernel_regularizer=l2(regularization_coef))(net)
        outputs.append(regr_output)
        sfmax_output = Dense(nr_output_classes, activation=softmax, name="Sfmax")
        outputs.append(sfmax_output(net))

    model = Model(inputs=input_, outputs=outputs)
    optimizer = optimizers.adam(lr=lr, clipvalue=clipvalue)
    model.compile(loss=model_loss, optimizer=optimizer, metrics=["accuracy"],
                  loss_weights=multiloss_weights if multiloss_weights is not None else 1)
    return model
