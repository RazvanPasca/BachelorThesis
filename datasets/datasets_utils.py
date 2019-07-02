from enum import Enum

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
    assert (nr_val_indices > 0)
    train_indices = shuffled_indices[:-nr_val_indices]
    val_indices = shuffled_indices[-nr_val_indices:]
    return (set(train_indices), set(val_indices)) if get_sets else (train_indices, val_indices)


class SequenceAddress():
    def __init__(self, split_location, movie, trial, channel, timestep, length):
        self.split_location = split_location
        self.movie = movie
        self.trial = trial
        self.channel = channel
        self.timestep = timestep
        self.length = length


class ModelType(Enum):
    NEXT_TIMESTEP = 1
    CONDITION_CLASSIFICATION = 2
    SCENE_CLASSIFICATION = 3
    BRIGHTNESS = 4
    EDGES = 5
    IMAGE_REC = 6


class SplitStrategy(Enum):
    TRIALS = 1
    SLICES = 2


class SlicingStrategy(Enum):
    RANDOM = 1
    CONSECUTIVE = 2
