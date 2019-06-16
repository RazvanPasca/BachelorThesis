import os

import numpy as np
from keras.backend import set_session, tf


def configure_gpu(gpu):
    global config
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    set_session(sess)


def replace_at_index(tup, ix, val):
    return tup[:ix] + (val,) + tup[ix + 1:]


def shuffle_indices(indices_nr, split_perc, get_sets):
    """

    Computes a permutation for indices_nr of indices and
    split them into two sets(or lists)

    """

    shuffled_indices = np.arange(indices_nr)
    np.random.shuffle(shuffled_indices)
    nr_val_indices = round(split_perc * indices_nr)
    train_indices = shuffled_indices[:-nr_val_indices]
    val_indices = shuffled_indices[-nr_val_indices:]
    return (set(train_indices), set(val_indices)) if get_sets else (train_indices, val_indices)
