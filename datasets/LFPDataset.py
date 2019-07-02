import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from datasets.datasets_utils import rescale, shuffle_indices, SequenceAddress, ModelType, SplitStrategy, SlicingStrategy
from signal_analysis.signal_utils import get_filter_type, butter_pass_filter, mu_law_fn
from utils.tf_utils import _split_dataset_into_slices

"""

Signal format:  (CONDITIONS, TRIALS, CHANNELS, TIMESTEPS)

Setups:
    NEXT_TIMESTEP    Next timestep generation - INPUT sequence(slice_length) - ACTUAL OUTPUT next timestep(bin) 
        SLICING_STRATEGY - RANDOM
        CHANNELS_IN_INPUT - SINGLE
        SPLIT - TRIALS
    MOVIE_CLASSIFICATION   Movie classification- INPUT sequence(slice_length) - ACTUAL OUTPUT class(one_hot) 
        SLICING_STRATEGY - RANDOM or CONSECUTIVE
        CHANNELS_IN_INPUT - SINGLE OR ALL
        SPLIT - TRIALS or SLICES
    # SCENE_CLASSIFICATION   Scene classification- INPUT sequence(slice_length) - ACTUAL OUTPUT class(one_hot) 
    #     SLICING_STRATEGY - RANDOM or CONSECUTIVE
    #     CHANNELS_IN_INPUT - SINGLE OR ALL
    #     SPLIT - TRIALS or SLICES
    # GAMMA_CONDITIONING   Gamma classification- INPUT sequence(slice_length) - ACTUAL OUTPUT class(one_hot) 
    #     SLICING_STRATEGY - RANDOM or CONSECUTIVE
    #     CHANNELS_IN_INPUT - SINGLE OR ALL
    #     SPLIT - TRIALS or SLICES
    BRIGHTNESS       Brightness regression - INPUT sequence(slice_length) - ACTUAL OUTPUT brightness values
        SLICING_STRATEGY - RANDOM or CONSECUTIVE
        CHANNELS_IN_INPUT - ALL
        SPLIT - TRIALS or SLICES
    EDGES            Edges regression - INPUT sequence(slice_length) - ACTUAL OUTPUT edges amount values
        SLICING_STRATEGY - RANDOM or CONSECUTIVE
        CHANNELS_IN_INPUT - ALL
        SPLIT - TRIALS or SLICES
    IMAGE_REC        Image Reconstruction - INPUT sequence(slice_length) - ACTUAL OUTPUT image
        SLICING_STRATEGY - RANDOM or CONSECUTIVE
        CHANNELS_IN_INPUT - ALL
        SPLIT - TRIALS or SLICES
"""


class LFPDataset:
    def __init__(self,
                 signal_path,
                 stimuli_path,
                 val_percentage,
                 split_by,
                 random_seed,
                 conditions_to_keep,
                 trials_to_keep,
                 channels_to_keep,
                 slice_length,
                 slicing_strategy,
                 model_type,
                 cutoff_freq,
                 stack_channels,
                 use_mu_law,
                 number_of_bins):
        np.random.seed(random_seed)

        self.slice_indexes = {}
        self.prepared_data = {}
        self.cached_bin_of_value = {}
        self.regression_target_mean = None
        self.regression_target_std = None
        self.number_of_channels = None
        self.stimuli = None
        self.signal = None

        self.random_seed = random_seed
        self.signal_path = signal_path
        self.stimuli_path = stimuli_path
        self.use_mu_law = use_mu_law
        self.nr_bins = number_of_bins
        self.val_percentage = val_percentage
        self.cutoff_freq = cutoff_freq
        self.stack_channels = stack_channels
        self.model_type = model_type
        self.slicing_strategy = slicing_strategy
        self.split_by = split_by
        self.slice_length = slice_length
        self.conditions_to_keep = np.array(conditions_to_keep) if type(conditions_to_keep) is list else None
        self.trials_to_keep = np.array(trials_to_keep) if type(trials_to_keep) is list else None
        self.channels_to_keep = np.array(channels_to_keep) if type(channels_to_keep) is list else None

        self._validate_setup()

        self._load_data()

        self._filter_data()

        self._prepare_data()

    def get_random_sequence(self, source):
        """

        Gets a random sequence from source

        """

        assert (self.slicing_strategy in [SlicingStrategy.CONSECUTIVE, SlicingStrategy.RANDOM])

        if self.slicing_strategy == SlicingStrategy.RANDOM:
            sequence, addr = self._get_random_sequence(source, return_address=True)
        elif self.slicing_strategy == SlicingStrategy.CONSECUTIVE:
            sequence, addr = self._get_random_slice(source, return_address=True)
        else:
            raise Exception("Invalid slicing strategy type!")

        return sequence, addr

    def get_random_example(self, source):
        """

        Gets a random example from source

        """

        sequence, seq_addr = self.get_random_sequence(source)
        label = self._get_y_value_for_sequence(seq_addr)

        return sequence, label

    def train_sample_generator(self, batch_size):
        """

        Returns the train generator of samples

        """

        return self._example_generator(batch_size, "TRAIN")

    def validation_sample_generator(self, batch_size):
        """

        Returns the validation generator of samples

        """

        return self._example_generator(batch_size, "VAL")

    def get_training_dataset_size(self):
        """

        Returns the number of samples in train

        """

        return self._get_dataset_size("TRAIN")

    def get_validation_dataset_size(self):
        """

        Returns the number of samples in validation

        """

        return self._get_dataset_size("VAL")

    def get_name(self):
        return type(self).__name__

    def _get_dataset_size(self, source):
        dimensions = list(self.prepared_data[source].shape)
        if self.stack_channels:
            dimensions[-2] = 1
        dimensions[-1] -= self.slice_length - 1
        size = 1
        for x in dimensions:
            size *= x
        return size

    def _prepare_data(self):
        self._filter_signal_frequencies()
        self._normalize_data()
        self._compute_values_range()

        if self.model_type == ModelType.EDGES:
            self._compute_stimuli_edges()
            self._compute_regression_target_stats()

        elif self.model_type == ModelType.BRIGHTNESS:
            self._compute_brightness_dataset()
            self._compute_regression_target_stats()

        elif self.model_type == ModelType.NEXT_TIMESTEP:
            self._pre_compute_bins()

        elif self.model_type == ModelType.CONDITION_CLASSIFICATION:
            self.nr_classes = len(self.conditions_to_keep)

        elif self.model_type == ModelType.SCENE_CLASSIFICATION:
            self.nr_classes = -1
            # TODO

        self._split_dataset()

    def _compute_values_range(self):
        """

        Computes the range between which signal is

        """

        assert (type(self.signal) is np.ndarray)

        self.min_val = np.min(self.signal)
        self.max_val = np.max(self.signal)

    def _pre_compute_bins(self):
        """

        Computes the bin size and the bins

        """

        assert (type(self.nr_bins) is int)
        assert (type(self.min_val) is np.float64)
        assert (type(self.max_val) is np.float64)

        self.bins = np.linspace(np.floor(self.min_val), np.ceil(self.max_val), self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]

    def _mu_law_encoding(self, value):
        """

        Translates the value to the mu law encoding bin
        taking into consideration the mu value
self.cached_bin_of_value[value]
        """

        assert (type(self.use_mu_law) is int)

        return np.rint(rescale(value, 1, -1, self.use_mu_law - 1, 0))

    def _encode_input_to_bin(self, value):
        """

        Maps value into a discretized bin

        """

        assert (type(self.cached_bin_of_value) is dict)
        assert (type(self.bins) is np.ndarray)

        if value not in self.cached_bin_of_value:
            if self.use_mu_law:
                self.cached_bin_of_value[value] = self._mu_law_encoding(value)
            else:
                self.cached_bin_of_value[value] = np.digitize([value], self.bins, right=False)[0]

        return self.cached_bin_of_value[value]

    def _split_dataset(self):
        """

        Splits dataset and prepares it for the generators

        """

        assert (type(self.val_percentage) is float)
        assert (type(self.signal) is np.ndarray)
        assert (self.slicing_strategy in [SlicingStrategy.CONSECUTIVE, SlicingStrategy.RANDOM])

        if self.slicing_strategy == SlicingStrategy.CONSECUTIVE:
            assert (self.split_by in [SplitStrategy.TRIALS, SplitStrategy.SLICES])

            nr_slices_per_channel = self.signal.shape[-1] // self.slice_length
            new_dataset = _split_dataset_into_slices(self.signal, self.slice_length)

            if self.split_by == SplitStrategy.TRIALS:
                train_indexes, val_indexes = shuffle_indices(self.trials_per_condition, self.val_percentage, False)
                self.slice_indexes["TRAIN"] = self.slice_indexes["VAL"] = np.arange(0, nr_slices_per_channel)
                self.prepared_data["VAL"] = new_dataset[:, val_indexes, ...]
                self.prepared_data["TRAIN"] = new_dataset[:, train_indexes, ...]

            elif self.split_by == SplitStrategy.SLICES:
                self.slice_indexes["TRAIN"], self.slice_indexes["VAL"] = shuffle_indices(nr_slices_per_channel,
                                                                                         self.val_percentage, False)
                self.prepared_data["VAL"] = new_dataset[:, :, :, self.slice_indexes["VAL"], :]
                self.prepared_data["TRAIN"] = new_dataset[:, :, :, self.slice_indexes["TRAIN"], :]

        elif self.slicing_strategy == SlicingStrategy.RANDOM:
            assert (self.split_by in [SplitStrategy.TRIALS])

            if self.split_by == SplitStrategy.TRIALS:
                self.train_indexes, self.val_indexes = shuffle_indices(self.trials_per_condition, self.val_percentage,
                                                                       False)
                self.prepared_data["VAL"] = self.signal[:, self.val_indexes, ...]
                self.prepared_data["TRAIN"] = self.signal[:, self.train_indexes, ...]

    def _example_generator(self, batch_size, source):
        """

        It is a generator that retrieves the input signal and the actual value.

        Signal base on the slicing strategy and actual value based on the model_output_type

        """

        assert (source in self.prepared_data)

        x = []
        y = []
        while 1:
            sequence, actual = self.get_random_example(source)

            x.append(sequence)
            y.append(actual)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def _get_random_sequence(self, source, return_address=False):
        """

        Gets an random slice from a random trial found in data
        with size of slice_length and the ground truth stimuli

        """
        assert (source in self.prepared_data)

        random_trial, (condition_index, trial_index) = self._get_random_trial(source)
        sequence_start_index = np.random.randint(low=0, high=random_trial.shape[1] - self.slice_length)

        if self.stack_channels:
            sequence = random_trial[:, sequence_start_index:sequence_start_index + self.slice_length]
            address = SequenceAddress(source, condition_index, trial_index, np.arange(random_trial.shape[0]),
                                      sequence_start_index,
                                      self.slice_length)
        else:
            channel = np.random.randint(low=0, high=self.number_of_channels)
            sequence = random_trial[channel,
                       sequence_start_index:sequence_start_index + self.slice_length][:, np.newaxis]
            address = SequenceAddress(source, condition_index, trial_index, channel, sequence_start_index,
                                      self.slice_length)

        result = [sequence]

        if return_address:
            result.append(address)

        return result

    def _get_random_slice(self, source, return_address=False):
        """

        Gets an random example from data with size of
        slice_length and the ground truth stimuli

        """

        assert (source in self.prepared_data)

        random_sequence, (movie_index, trial_index) = self._get_random_trial(source)
        slice_index = np.random.randint(low=0, high=random_sequence.shape[1])
        signal_sequence = np.transpose(random_sequence[:, slice_index, :])

        timestep = self.slice_indexes[source][slice_index] * self.slice_length
        address = SequenceAddress(source, movie_index, trial_index, ":", timestep, self.slice_length)

        result = [signal_sequence]

        if return_address:
            result.append(address)

        return result

    def _get_random_trial(self, source):
        """

        Gets a random trial from data_source

        """

        assert (source in self.prepared_data)

        number_of_conditions = self.prepared_data[source].shape[0]
        number_of_trials = self.prepared_data[source].shape[1]

        condition_index = np.random.choice(self.prepared_data[source].shape[0]) if number_of_conditions > 1 else 0
        trial_index = np.random.choice(self.prepared_data[source].shape[1]) if number_of_trials > 1 else 0

        condition = self.prepared_data[source][condition_index] if number_of_conditions > 1 else self.prepared_data[
            source]
        trial = condition[trial_index] if number_of_trials > 1 else condition

        return trial, (condition_index, trial_index)

    def _load_data(self):
        """

        Loads the data into self.signal and self.stimuli
        and pre-processes the data

        """

        self.signal = np.load(self.signal_path)
        self.number_of_conditions = self.signal.shape[0]
        self.trials_per_condition = self.signal.shape[1]
        self.number_of_channels = self.signal.shape[-2]
        self.trial_length = self.signal.shape[-1]

        if os.path.exists(self.stimuli_path):
            self.stimuli = np.load(self.stimuli_path)
            self.stimuli_width = self.signal.shape[0]
            self.stimuli_height = self.signal.shape[1]

    def _compute_brightness_dataset(self):
        """

        Computes brightness from stimuli

        """

        self.regression_actual = np.mean(self.stimuli, axis=(2, 3))

    def _compute_stimuli_edges(self):
        """

        Computes edges from stimuli

        """

        stimuli_w_edges_extracted = np.zeros(self.stimuli.shape)
        smoothed_stimuli = np.zeros(self.stimuli.shape)
        kernel = np.ones((1, 1), np.uint8)
        morph_kernel = np.ones((3, 3), np.float32) / 9

        for i, movie in enumerate(self.stimuli):
            for j, image in enumerate(movie):
                # image = cv.filter2D(image, -1, kernel)
                highThreshold = np.max(image) * 0.7
                lowThreshold = highThreshold * 0.7
                edges = cv.Canny((image * 255).astype(np.uint8), lowThreshold, highThreshold, L2gradient=True)
                # edges = cv.morphologyEx(edges, cv.MORPH_OPEN, morph_kernel)
                # kernel = np.ones((1, 2), np.uint8)
                # edges = cv.morphologyEx(edges, cv.MORPH_ERODE, kernel)
                stimuli_w_edges_extracted[i, j, :, :] = edges
                smoothed_stimuli[i, j, :, :] = image

        self.regression_actual = np.mean(stimuli_w_edges_extracted, axis=(2, 3)) / 255
        self.stimuli_w_edges_extracted = stimuli_w_edges_extracted

    def _get_y_value_for_sequence(self, seq_addr: SequenceAddress):
        """

        Finds the a sequence based on the address

        """
        if self.model_type == ModelType.CONDITION_CLASSIFICATION:
            return [seq_addr.movie]

        elif self.model_type == ModelType.SCENE_CLASSIFICATION:
            # TODO
            pass

        elif self.model_type == ModelType.NEXT_TIMESTEP:
            next_timestep = self.prepared_data[seq_addr.split_location][
                seq_addr.movie, seq_addr.trial, seq_addr.channel, seq_addr.timestep + self.slice_length]
            return [self._encode_input_to_bin(next_timestep)]

        else:
            image_number = (seq_addr.timestep - 100) // 40
            if self.model_type == ModelType.IMAGE_REC:
                image = self.stimuli[seq_addr.movie, image_number, :, :]
                return image[:, :, np.newaxis]
            elif self.model_type == ModelType.BRIGHTNESS or self.model_type == ModelType.EDGES:
                return [self.regression_actual[seq_addr.movie, image_number]]

            raise Exception("Invalid ModelType!")

    def get_nr_classes(self):
        """

        Returns the number of classes

        """

        if self.model_type == ModelType.CONDITION_CLASSIFICATION:
            return self.number_of_conditions
        elif self.model_type == ModelType.SCENE_CLASSIFICATION:
            raise Exception("Not implemented!")

    def _compute_regression_target_stats(self):
        """

        Computes regression target mean and std dev

        """

        self.regression_target_mean = np.mean(self.regression_actual)
        self.regression_target_std = np.std(self.regression_actual)

    def _validate_setup(self):
        assert os.path.exists(self.signal_path)

        if self.model_type == ModelType.NEXT_TIMESTEP:
            assert (self.stack_channels == False)

        if self.slicing_strategy == SlicingStrategy.RANDOM:
            assert (self.split_by == SplitStrategy.TRIALS)

        if self.model_type == ModelType.NEXT_TIMESTEP:
            assert (self.slicing_strategy == SlicingStrategy.RANDOM)
            assert (self.split_by == SplitStrategy.TRIALS)
        # TODO complete this

    def _filter_data(self):
        if self.conditions_to_keep is not None:
            self.signal = self.signal[self.conditions_to_keep, ...]
            self.number_of_conditions = self.signal.shape[0]
        if self.trials_to_keep is not None:
            self.signal = self.signal[:, self.trials_to_keep, ...]
            self.trials_per_condition = self.signal.shape[-3]
        if self.channels_to_keep is not None:
            self.signal = self.signal[:, :, self.conditions_to_keep, ...]
            self.number_of_channels = self.signal.shape[-2]

    def _filter_signal_frequencies(self):
        filter_type = get_filter_type(self.cutoff_freq)
        if filter_type is not None:
            self.signal = butter_pass_filter(self.signal, self.cutoff_freq, 1000, filter_type)

    def _normalize_data(self):
        """

        Normalizes the data

        """

        assert (type(self.signal) is np.ndarray)

        self._normalize_signal_per_channel()
        if self.stimuli is not None:
            self._normalize_stimuli()

    def _normalize_signal_per_channel(self):
        nr_channels = self.signal.shape[2]

        if self.use_mu_law:
            """

            The signal is brought to [-1,1] through rescale->[-1,1] mu_law and then encoded using np.digitize

            """
            self.limits = {}
            self.mu_law = True

            for i in range(nr_channels):
                np_min = np.min(self.signal[:, :, i, :], keepdims=True)
                np_max = np.max(self.signal[:, :, i, :], keepdims=True)
                self.limits[i] = (np_min, np_max)
                self.signal[:, :, i, :] = mu_law_fn(
                    rescale(self.signal[:, :, i, :], old_max=np_max, old_min=np_min, new_max=1, new_min=-1),
                    self.nr_bins)

        else:
            for i in range(nr_channels):
                mean = np.mean(self.signal[:, :, i, :], keepdims=True)
                std = np.std(self.signal[:, :, i, :], keepdims=True)
                self.signal[:, :, i, :] -= mean
                self.signal[:, :, i, :] /= std

    def _normalize_stimuli(self):
        self.stimuli = self.stimuli / np.max(self.stimuli)
        self.stimuli = self.stimuli - np.min(self.stimuli)

    def get_input_shape(self):
        x, _ = self.get_random_example("TRAIN")
        return x.shape


def show_edges_computed():
    movies_keep = [0, 1, 2]
    dataset = LFPDataset(val_percentage=0.15, model_type="EDGES", conditions_to_keep=movies_keep,
                         split_by=SplitStrategy.SLICES)
    np.random.seed(42)
    for mov in movies_keep:
        for j in np.random.choice(700, 10):
            plt.subplot(121), plt.imshow(dataset.stimuli[mov, j], cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dataset.stimuli_w_edges_extracted[mov, j], cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(133),
            # plt.imshow(dataset.smoothed_stimuli[movie, j], cmap='gray')
            # plt.title('Smoothed Image'), plt.xticks([]), plt.yticks([])
            plt.show()


if __name__ == '__main__':
    show_edges_computed()
