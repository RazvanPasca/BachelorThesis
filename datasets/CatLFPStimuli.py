import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from datasets.dataset_utils import rescale, shuffle_indices
from datasets.paths import CAT_DATASET_SIGNAL_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH_64
from signal_analysis.signal_utils import get_filter_type, filter_input_sample


class CatLFPStimuli:
    def __init__(self,
                 movies_to_keep=[0, 1, 2],
                 cutoff_freq=None,
                 val_perc=0.20,
                 random_seed=42,
                 model_output_type="DCGAN",
                 split_by="trials",
                 slice_length=1000,
                 slicing_strategy="consecutive"):
        np.random.seed(random_seed)

        self.cached_bin_of_value = {}
        self.cutoff_freq = cutoff_freq
        self.model_output_type = model_output_type.upper()
        self.classification_type = self._get_classification_type(model_output_type)
        self.slicing_strategy = slicing_strategy.upper()
        self.split_by = split_by.upper()
        self.slice_length = slice_length

        self.movies_to_keep = np.array(movies_to_keep)
        self._load_data(CAT_DATASET_SIGNAL_PATH, CAT_DATASET_STIMULI_PATH_64, CAT_DATASET_STIMULI_PATH)
        self.number_of_channels = self.signal.shape[-2]

        self._compute_values_range()
        self._pre_compute_bins()
        self._normalize_data()
        self.nr_classes = self._get_nr_classes()
        self.nr_conditions = self.signal.shape[0]
        self.trials_per_condition = self.signal.shape[1]
        self._split_dataset(val_perc)

    def _compute_values_range(self):
        """

        Computes the range between which signal is

        """
        assert(type(self.signal) is np.ndarray)

        self.min_val = np.min(self.signal)
        self.max_val = np.max(self.signal)

    def _pre_compute_bins(self):
        """

        Computes the bin size and the bins

        """
        assert(type(self.nr_bins) is int)
        assert(type(self.min_val) is int)
        assert(type(self.max_val) is int)

        self.bins = np.linspace(np.floor(self.min_val), np.ceil(self.max_val), self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]

    def _mu_law_encoding(self, value):
        """

        Translates the value to the mu law encoding bin
        taking into consideration the mu value

        """
        assert (type(self.mu_value) is int)

        return np.rint(rescale(value, 1, -1, self.mu_value - 1, 0))

    def _encode_input_to_bin(self, value):
        """

        Maps value into a discretized bin

        """
        assert(type(self.cached_bin_of_value) is dict)
        assert(type(self.bins) is np.ndarray)

        if value not in self.cached_bin_of_value:
            if self.mu_law_activated:
                self.cached_bin_of_value[value] = self._mu_law_encoding(value)
            else:
                self.cached_bin_of_value[value] = np.digitize(value, self.bins, right=False)

        return self.cached_bin_of_value[value]

    def _split_dataset(self):
        """

        Splits dataset and prepares it for the generators

        """
        assert(type(self.val_percentage) is int)

        self.signal = self.signal[self.movies_to_keep, ...]

        if self.slicing_strategy == "CONSECUTIVE":
            nr_slices_per_channel = self.signal.shape[-1] // self.slice_length
            new_dataset = self.get_dataset_with_slices()

            if self.split_by == "TRIALS":
                train_indexes, val_indexes = shuffle_indices(self.trials_per_condition, self.val_percentage, False)
                self.train_slices = self.val_slices = np.arange(0, nr_slices_per_channel)
                self.validation = new_dataset[:, val_indexes, ...]
                self.train = new_dataset[:, train_indexes, ...]

            elif self.split_by == "SLICES":
                self.train_slices, self.val_slices = shuffle_indices(nr_slices_per_channel, self.val_percentage, False)
                self.validation = new_dataset[:, :, :, self.val_slices, :]
                self.train = new_dataset[:, :, :, self.train_slices, :]

        elif self.slicing_strategy == "RANDOM":
            # not implemented
            pass
            self.train_indexes, self.val_indexes = shuffle_indices(self.trials_per_condition, self.val_percentage, False)
            self.validation = self.signal[:, self.val_indexes, ...]
            self.train = self.signal[:, self.train_indexes, ...]

    def get_dataset_with_slices(self):
        """

        Slices dataset into another dimension based on the slice_length

        """

        nr_slices_per_channel = self.signal.shape[-1] // self.slice_length
        new_dataset = np.zeros((self.signal.shape[:-1] + (nr_slices_per_channel, self.slice_length)))
        for movie in range(self.signal.shape[0]):
            for trial in range(self.signal.shape[1]):
                slices = np.zeros((nr_slices_per_channel, self.slice_length))
                for channel in range(self.signal.shape[2]):
                    for i in range(nr_slices_per_channel):
                        slices[i] = self.signal[movie, trial, channel,
                                    i * self.slice_length: (i + 1) * self.slice_length]
                    new_dataset[movie, trial, channel] = slices

        return new_dataset

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

    def train_sample_generator(self, frame_size, batch_size, with_stimuli=True):
        """

        Returns the train generator of samples

        """

        if with_stimuli:
            return self.generator_of_sample_with_stimuli(frame_size, batch_size, self.train, "train")
        else:
            # TODO
            pass

    def validation_sample_generator(self, frame_size, batch_size, with_stimuli=True):
        """

        Returns the validation generator of samples

        """

        if with_stimuli:
            return self.generator_of_sample_with_stimuli(frame_size, batch_size, self.validation, "val")
        else:
            # TODO
            pass

    def _get_random_frame_stimuli_slices(self, data, source):
        """

        Gets an random example from data with size of
        slice_length and the ground truth stimuli

        """

        random_sequence, (movie_index, trial_index) = self.get_random_sequence(data)
        slice_index = np.random.randint(low=0, high=random_sequence.shape[1])
        signal_sequence = random_sequence[:, slice_index, :]
        if source.lower() == "val":
            timestamp = self.val_slices[slice_index] * self.slice_length
        else:
            timestamp = self.train_slices[slice_index] * self.slice_length
        ground_truth_stimuli = self._get_stimuli_for_sequence(movie_index, timestamp)

        return signal_sequence, ground_truth_stimuli

    def get_random_sequence(self, data_source):
        """

        Gets a random trial from data_source

        """

        movie_index = np.random.choice(data_source.shape[0])
        trial_index = np.random.choice(data_source.shape[1])
        return data_source[movie_index, trial_index], (movie_index, trial_index)

    def _load_data(self, signal_path, stimuli_path):
        """

        Loads the data into self.signal and self.stimuli
        and pre-processes the data

        """

        self.signal = np.load(signal_path)[self.movies_to_keep, ...]
        filter_type = get_filter_type(self.cutoff_freq)
        self.signal = filter_input_sample(self.signal, self.cutoff_freq, filter_type)
        self.stimuli = np.load(stimuli_path)[self.movies_to_keep, ...]

        if self.model_output_type == "EDGES":
            self.stimuli_regression_target, self.stimuli_w_edges_extracted = self.extract_edges_dataset(self.stimuli)
            self.get_regression_target_stats()
        elif self.model_output_type == "BRIGHTNESS":
            self.stimuli_regression_target = self.extract_brightness_dataset(self.stimuli)
            self.get_regression_target_stats()

    def extract_brightness_dataset(self, stimuli):
        """

        Computes brightness from stimuli

        """

        stimuli_regression_target_ = np.mean(stimuli / 255, axis=(2, 3))
        return stimuli_regression_target_

    def extract_edges_dataset(self, stimuli):
        """

        Computes edges from stimuli

        """

        stimuli_w_edges_extracted = np.zeros(stimuli.shape)
        smoothed_stimuli = np.zeros(stimuli.shape)
        kernel = np.ones((1, 1), np.uint8)
        morph_kernel = np.ones((3, 3), np.float32) / 9

        for i, movie in enumerate(stimuli):
            for j, image in enumerate(movie):
                # image = cv.filter2D(image, -1, kernel)
                highThreshold = np.max(image) * 0.7
                lowThreshold = highThreshold * 0.7
                edges = cv.Canny(image, lowThreshold, highThreshold, L2gradient=True)
                # edges = cv.morphologyEx(edges, cv.MORPH_OPEN, morph_kernel)
                # kernel = np.ones((1, 2), np.uint8)
                # edges = cv.morphologyEx(edges, cv.MORPH_ERODE, kernel)
                stimuli_w_edges_extracted[i, j, :, :] = edges
                smoothed_stimuli[i, j, :, :] = image

        stimuli_regression_target = np.mean(stimuli_w_edges_extracted, axis=(2, 3)) / 255
        return stimuli_regression_target, stimuli_w_edges_extracted

    def _get_stimuli_for_sequence(self, movie_index, seq_start):
        """

        Finds the stimuli for a sequence based on movie index and seq_start

        """

        image_number = (seq_start - 100) // 40
        if self.model_output_type == "DCGAN":
            image = self.stimuli[movie_index, image_number, :, :]
            return image[:, :, np.newaxis]
        elif self.model_output_type == "BRIGHTNESS" or self.model_output_type == "EDGES":
            return [self.stimuli_regression_target[movie_index, image_number]]
        elif self.model_output_type == "CLASSIFY_MOVIES":
            return [movie_index]

    # TODO - fix it
    def _normalize_data(self):
        """

        Normalizes the data

        """

        for channel in range(self.signal.shape[2]):
            self.signal[:, :, channel, :] /= np.max(self.signal[:, :, channel, :])
        self.stimuli = self.stimuli / np.max(self.stimuli)

    def _get_classification_type(self, classification_type):
        """

        Gets the classification type

        """

        if classification_type.find("movies"):
            return "movies"
        elif classification_type.find("scenes"):
            return "scenes"

    def _get_nr_classes(self):
        """

        Returns the number of classes

        """

        if self.classification_type == "movies":
            return len(self.movies_to_keep)

    def get_regression_target_stats(self):
        """

        Computes regression target mean and std dev

        """

        self.regression_target_mean = np.mean(self.stimuli_regression_target)
        self.regression_target_std = np.std(self.stimuli_regression_target)

    def plot_stimuli_hist(self, path):
        """

        Plots stimuli histogram.

        """

        plt.figure(figsize=(16, 12))
        plt.hist([movie for movie in self.stimuli_regression_target], bins=30,
                 label=["Movie:{}".format(i) for i in self.movies_to_keep])
        plt.legend(prop={'size': 10})
        plt.title("Histogram of {} with mean {:10.4f} and std {:10.4f}".format(self.model_output_type,
                                                                               self.regression_target_mean,
                                                                               self.regression_target_std))
        plt.savefig("{}/movies_regression_histo.png".format(path), format="png")


if __name__ == '__main__':
    movies_to_keep = [0, 1, 2]
    dataset = CatLFPStimuli(val_perc=0.15, model_output_type="EDGES", movies_to_keep=movies_to_keep, split_by="SLICES")
    np.random.seed(42)
    for movie in movies_to_keep:
        for j in np.random.choice(700, 10):
            plt.subplot(121), plt.imshow(dataset.stimuli[movie, j], cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(dataset.stimuli_w_edges_extracted[movie, j], cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            # plt.subplot(133),
            # plt.imshow(dataset.smoothed_stimuli[movie, j], cmap='gray')
            # plt.title('Smoothed Image'), plt.xticks([]), plt.yticks([])
            plt.show()
