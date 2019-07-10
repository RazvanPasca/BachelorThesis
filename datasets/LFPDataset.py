import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from datasets.datasets_utils import rescale, shuffle_indices, SequenceAddress, ModelType, SplitStrategy, SlicingStrategy
from datasets.paths import CAT_DATASET_STIMULI_PATH_64, CAT_DATASET_SIGNAL_PATH
from signal_analysis.signal_utils import mu_law_fn, filter_input_sequence
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
                 number_of_bins,
                 condition_on_gamma,
                 gamma_windows_in_trial,
                 blur_images,
                 relative_difference):
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
        self.condition_on_gamma = condition_on_gamma
        self.blur_images = blur_images
        self.relative_difference = relative_difference

        self.conditions_to_keep = np.array(conditions_to_keep) if type(conditions_to_keep) is list else None
        self.trials_to_keep = np.array(trials_to_keep) if type(trials_to_keep) is list else None
        self.channels_to_keep = np.array(channels_to_keep) if type(channels_to_keep) is list else None

        self._validate_setup()

        self._load_data()

        self._filter_data()

        self._prepare_data()

        self.gamma_windows_in_trial = [item for sublist in gamma_windows_in_trial for item in sublist if
                                       item < self.trial_length]

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

        return sequence, seq_addr, label

    def train_sample_generator(self, batch_size, return_address=False):
        """

        Returns the train generator of samples
        :param return_address: whether a list with the addresses for each sample should be returned or not

        """

        return self._example_generator(batch_size, "TRAIN", return_address)

    def validation_sample_generator(self, batch_size, return_address=False):
        """

        Returns the validation generator of samples
        :param return_address: whether a list with the addresses for each sample should be returned or not

        """

        return self._example_generator(batch_size, "VAL", return_address)

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
        size = self.prepared_data[source].size
        if self.stack_channels:
            size = size // self.number_of_channels
        return size

    def _prepare_data(self):
        self._filter_signal_frequencies()
        self._normalize_data()
        self._compute_values_range()

        if self.blur_images and self.model_type in [ModelType.IMAGE_REC, ModelType.EDGES]:
            self._blur_stimuli()

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
            class_counter = 0
            self.classes_label = {}
            for key in self.conditions_to_keep:
                self.classes_label[key] = class_counter
                class_counter += 1

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

        return np.rint(rescale(value, 1, -1, self.use_mu_law - 1, 0))

    def _encode_input_to_bin(self, value):
        """

        Maps value into a discretized bin

        """

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
                # TO DO for each trial use a different set of slices, otherwise we will never see the first slice                  #for any trial  in train/test which might be bad
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

    def _example_generator(self, batch_size, source, return_address=False):
        """

        It is a generator that retrieves the input signal and the y value.

        Signal base on the slicing strategy and actual value based on the model_output_type
        :param return_address:

        """

        x = []
        y = []
        adresses = []
        while 1:
            sequence, seq_address, actual = self.get_random_example(source)

            x.append(sequence)
            y.append(actual)
            adresses.append(seq_address)
            if len(x) == batch_size:
                if return_address:
                    yield np.array(x), np.array(y), adresses
                else:
                    yield np.array(x), np.array(y)
                x = []
                y = []

    def _get_random_sequence(self, source, return_address=False):
        """

        Gets an random slice from a random trial found in source
        with size of slice_length and the ground truth stimuli

        """

        random_trial, (condition_index, trial_index) = self._get_random_trial(source)
        sequence_start_index = np.random.randint(low=0, high=random_trial.shape[1] - self.slice_length)

        if self.stack_channels:
            sequence = random_trial[:, sequence_start_index:sequence_start_index + self.slice_length]
            address = SequenceAddress(condition_index, trial_index, np.arange(random_trial.shape[0]),
                                      sequence_start_index, self.slice_length, source)
        else:
            channel = np.random.randint(low=0, high=self.number_of_channels)
            sequence = random_trial[channel,
                       sequence_start_index:sequence_start_index + self.slice_length][:, np.newaxis]
            address = SequenceAddress(condition_index, trial_index, channel, sequence_start_index, self.slice_length,
                                      source)

        result = [sequence]

        if return_address:
            result.append(address)

        return result

    def _get_random_slice(self, source, return_address=False):
        """

        Gets an random example from data with size of
        slice_length and the ground truth stimuli

        """

        random_sequence, (movie_index, trial_index) = self._get_random_trial(source)
        slice_index = np.random.randint(low=0, high=random_sequence.shape[1])
        timestep = self.slice_indexes[source][slice_index] * self.slice_length

        if self.stack_channels:
            signal_sequence = np.transpose(random_sequence[:, slice_index, :])
            address = SequenceAddress(movie_index, trial_index, ":", timestep, self.slice_length, source)

        else:
            channel = np.random.randint(low=0, high=self.number_of_channels)
            signal_sequence = random_sequence[channel, slice_index, :][:, np.newaxis]
            address = SequenceAddress(movie_index, trial_index, channel, timestep, self.slice_length, source)

        result = [signal_sequence]

        if return_address:
            result.append(address)

        return result

    def _get_random_trial(self, source):
        """

        Gets a random trial from data_source

        """

        condition_index = np.random.choice(self.prepared_data[source].shape[0])
        trial_index = np.random.choice(self.prepared_data[source].shape[1])

        condition = self.prepared_data[source][condition_index]
        trial = condition[trial_index, ...]

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
            self.stimuli_width = self.stimuli.shape[2]
            self.stimuli_height = self.stimuli.shape[3]
            self.stimuli_depth = 1

    def _compute_brightness_dataset(self):
        """

        Computes brightness from stimuli

        """

        self.regression_actual = np.mean(self.stimuli, axis=(2, 3))
        if self.relative_difference:
            relative_regression_actual = np.zeros(self.regression_actual.shape)
            for i, cond in enumerate(self.regression_actual):
                for j, regression_actual in enumerate(cond):
                    relative_regression_actual[i, j] = regression_actual - cond[j - 1] if j > 0 else \
                        regression_actual
            self.regression_actual = rescale(relative_regression_actual, relative_regression_actual.max(),
                                             relative_regression_actual.min(), 1, -1)

    def _compute_stimuli_edges(self):
        """
r
        Computes edges from stimuli

        """

        # cannyDetector = CannyEdgeDetector.CannyEdgeDetector([])

        stimuli_w_edges_extracted = np.zeros(self.stimuli.shape)

        for i, movie in enumerate(self.stimuli):
            for j, image in enumerate(movie):
                int_image = image * 255
                int_image = np.uint8(int_image)
                int_image = cv.bilateralFilter(int_image, 3, 50, 50)
                # edge_image = cannyDetector.detect(image)
                highThreshold = np.max(int_image) * 0.6
                lowThreshold = highThreshold * 0.6
                edge_image = cv.Canny(np.uint8(int_image), lowThreshold, highThreshold, L2gradient=True)
                stimuli_w_edges_extracted[i, j, :, :] = edge_image

        self.regression_actual = np.mean(stimuli_w_edges_extracted, axis=(2, 3)) / 255
        if self.relative_difference:
            relative_regression_actual = np.zeros(self.regression_actual.shape)
            for i, cond in enumerate(self.regression_actual):
                for j, regression_actual in enumerate(cond):
                    relative_regression_actual[i, j] = regression_actual - cond[j - 1] if j > 0 else \
                        regression_actual
            self.regression_actual = rescale(relative_regression_actual, relative_regression_actual.max,
                                             relative_regression_actual.min, 1, -1)

        self.stimuli_w_edges_extracted = stimuli_w_edges_extracted

    def _get_y_value_for_sequence(self, seq_addr: SequenceAddress):
        """

        Finds the a sequence based on the address

        """

        if self.model_type == ModelType.CONDITION_CLASSIFICATION:
            return self._get_condition(seq_addr)

        elif self.model_type == ModelType.SCENE_CLASSIFICATION:
            # TODO
            pass

        elif self.model_type == ModelType.NEXT_TIMESTEP:
            next_timestep = self.prepared_data[seq_addr.source] \
                [seq_addr.condition, seq_addr.trial, seq_addr.channel, seq_addr.timestep + self.slice_length]
            return [self._encode_input_to_bin(next_timestep)]

        else:
            image_number = (seq_addr.timestep - 100) // 40
            if self.model_type == ModelType.IMAGE_REC:
                image = self.stimuli[seq_addr.condition, image_number, :, :]
                return image[:, :, np.newaxis]
            elif self.model_type == ModelType.BRIGHTNESS or self.model_type == ModelType.EDGES:
                return [self.regression_actual[seq_addr.condition, image_number]]

            raise Exception("Invalid ModelType!")

    def _get_condition(self, seq_addr):
        return [seq_addr.movie]

    def get_nr_classes(self):
        """

        Returns the number of classes

        """

        if self.model_type == ModelType.CONDITION_CLASSIFICATION:
            return 8  # self.number_of_conditions
        elif self.model_type == ModelType.SCENE_CLASSIFICATION:
            raise Exception("Not implemented!")

    def _compute_regression_target_stats(self):
        """

        Computes regression target mean and std dev

        """

        self.regression_target_mean = [np.mean(self.regression_actual)]

        condition_means = np.mean(self.regression_actual, axis=1).flatten()
        for x in condition_means:
            self.regression_target_mean.append(x)

        condition_std = np.std(self.regression_actual, axis=1).flatten()
        self.regression_target_std = [np.std(self.regression_actual)]
        for x in condition_std:
            self.regression_target_std.append(x)

    def _validate_setup(self):
        assert os.path.exists(self.signal_path)

        if self.model_type == ModelType.NEXT_TIMESTEP:
            assert (self.stack_channels is False)

        if self.slicing_strategy == SlicingStrategy.RANDOM:
            assert (self.split_by == SplitStrategy.TRIALS)

        if self.model_type == ModelType.NEXT_TIMESTEP:
            assert (self.slicing_strategy == SlicingStrategy.RANDOM)
            assert (self.split_by == SplitStrategy.TRIALS)
        # TODO complete this

    def _filter_data(self):
        if self.conditions_to_keep is not None:
            self.signal = self.signal[self.conditions_to_keep, ...]
            self.stimuli = self.stimuli[self.conditions_to_keep, ...]
            self.number_of_conditions = self.signal.shape[0]
        else:
            self.conditions_to_keep = np.arange(self.signal.shape[0])
        if self.trials_to_keep is not None:
            self.signal = self.signal[:, self.trials_to_keep, ...]
            self.trials_per_condition = self.signal.shape[-3]
        if self.channels_to_keep is not None:
            self.signal = self.signal[:, :, self.channels_to_keep, ...]
            self.number_of_channels = self.signal.shape[-2]

    def _filter_signal_frequencies(self):
        self.signal = filter_input_sequence(self.signal, self.cutoff_freq)

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
        x, _, _ = self.get_random_example("TRAIN")

        return x.shape

    def get_train_plot_examples(self, nr_examples):
        if self.model_type != ModelType.NEXT_TIMESTEP:
            generator = self.train_sample_generator(nr_examples, True)
            self.train_examples = next(generator)
            generator.close()
        else:
            self.train_examples = self._get_next_timestep_examples(nr_examples, "TRAIN")
        return self.train_examples

    def get_val_plot_examples(self, nr_examples):
        if self.model_type != ModelType.NEXT_TIMESTEP:
            generator = self.validation_sample_generator(nr_examples, True)
            self.val_examples = next(generator)
            generator.close()
        else:
            self.val_examples = self._get_next_timestep_examples(nr_examples, "VAL")

        return self.val_examples

    def _get_next_timestep_examples(self, nr_examples, source):

        sequences = [None] * nr_examples
        addresses = [None] * nr_examples
        channels = np.random.randint(low=0, high=self.number_of_channels, size=nr_examples)

        for i, index in enumerate(channels):
            random_trial, (condition_index, trial_index) = self._get_random_trial(source)

            sequence = random_trial[index][:, np.newaxis]
            sequences[i] = sequence
            address = SequenceAddress(condition_index, trial_index, index, 0, self.trial_length, source)
            addresses[i] = address

        return sequences, addresses

    def _blur_stimuli(self):
        size = 15
        sigma = 3
        kernel = np.fromfunction(lambda x, y: (1 / (2 * np.math.pi * sigma ** 2)) * np.math.e ** (
                (-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)), (size, size))
        for i, movie in enumerate(self.stimuli):
            for j, image in enumerate(movie):
                new_image = cv.filter2D(image, -1, kernel)
                self.stimuli[i, j, :, :] = new_image


def show_edges_computed(dataset):
    movies_keep = [0, 1, 2]

    np.random.seed(42)
    for mov in movies_keep:
        for j in np.random.choice(700, 10):
            plt.subplot(121), plt.imshow(dataset.stimuli[mov, j], cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dataset.stimuli_w_edges_extracted[mov, j], cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()


def stimuli_histogram(dataset):
    plt.figure(figsize=(16, 12))
    plt.hist([movie for movie in dataset.regression_actual], bins=25,
             label=["Condition:{0:.0f}-Mean:{1:.4f}-Std:{2:.4f}".format(i, dataset.regression_target_mean[i + 1],
                                                                        dataset.regression_target_std[i + 1]) for i in
                    dataset.conditions_to_keep])
    plt.legend(fontsize=20)
    plt.title(
        "Overall-Mean:{0:.4f} Overall-Std:{1:.4f}".format(dataset.regression_target_mean[0],
                                                          dataset.regression_target_std[0]),
        fontsize=25)
    plt.xlabel("Stiumulus brightness value", fontsize=20)
    plt.ylabel("Nr stimuli", fontsize=20)
    plt.show()


if __name__ == '__main__':
    from training_parameters import training_parameters

    dataset_args = training_parameters["dataset_args"]
    dataset = LFPDataset(signal_path=CAT_DATASET_SIGNAL_PATH, stimuli_path=CAT_DATASET_STIMULI_PATH_64, **dataset_args)
    # stimuli_histogram(dataset)
    # show_edges_computed(dataset)

    print("MEAN: ", dataset.regression_target_mean)
    print("STD: ", dataset.regression_target_std)
