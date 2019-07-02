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


def generator_of_sample_with_stimuli(self, frame_size, batch_size, data, source):
    """

    It is a generator that retrieves signal and stimuli.

    Signal base on the slicing strategy and stimuli based on the model_output_type

    """

    x = []
    y = []
    while 1:
        frame, stimuli = self.get_random_sample_with_stimuli(data, frame_size, source)

        x.append(frame.transpose())
        y.append(stimuli)
        if len(x) == batch_size:
            yield np.array(x), np.array(y)
            x = []
            y = []


def get_random_sample_with_stimuli(self, data, frame_size, source):
    """

    Gets a random sample using the slicing strategy attribute

    """

    if self.slicing_strategy == "RANDOM":
        # not implemented
        pass
        frame, stimuli = self._get_random_frame_stimuli_trials(frame_size, data)
    else:
        frame, stimuli = self._get_random_frame_stimuli_slices(data, source)
    return frame, stimuli


def _get_random_frame_stimuli_slices(self, data, source):
    """

    Gets an random example from data with size of
    slice_length and the ground truth stimuli

    """

    random_sequence, (movie_index, trial_index) = self._get_random_trial(data)
    slice_index = np.random.randint(low=0, high=random_sequence.shape[1])
    signal_sequence = random_sequence[:, slice_index, :]
    if source.lower() == "val":
        timestamp = self.val_slices[slice_index] * self.slice_length
    else:
        timestamp = self.slice_indexes["TRAIN"][slice_index] * self.slice_length
    ground_truth_stimuli = self._get_y_value_for_sequence(movie_index, timestamp)

    return signal_sequence, ground_truth_stimuli


def _split_dataset_into_slices(signal, slice_length):
    """

    Slices dataset into another dimension based on the slice_length
    and returns it

    """

    nr_slices_per_channel = signal.shape[-1] // slice_length
    new_dataset = np.zeros((signal.shape[:-1] + (nr_slices_per_channel, slice_length)))
    for movie in range(signal.shape[0]):
        for trial in range(signal.shape[1]):
            slices = np.zeros((nr_slices_per_channel, slice_length))
            for channel in range(signal.shape[2]):
                for i in range(nr_slices_per_channel):
                    slices[i] = signal[movie, trial, channel,
                                i * slice_length: (i + 1) * slice_length]
                new_dataset[movie, trial, channel] = slices

    return new_dataset
