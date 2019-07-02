from keras import Input
from keras.layers import Conv1D, Multiply, Add, Activation, Flatten, Dense, Reshape, Lambda, BatchNormalization, \
    Conv2D, LeakyReLU, UpSampling2D
from keras.regularizers import l2


def get_wavenet_encoder():
    pass