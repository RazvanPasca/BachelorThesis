import numpy as np


def rescale(value, old_max, old_min, new_max, new_min):
    """

    Rescales the value from old range to new range

    """

    return (new_max - new_min) * (value - old_min) / (old_max - old_min) + new_min


def shuffle_indices(indices_nr, split_perc, get_sets):
    shuffled_indices = np.arange(indices_nr)
    np.random.shuffle(shuffled_indices)
    nr_val_indices = round(split_perc * indices_nr)
    train_indices = shuffled_indices[:-nr_val_indices]
    val_indices = shuffled_indices[-nr_val_indices:]
    return (set(train_indices), set(val_indices)) if get_sets else (train_indices, val_indices)
