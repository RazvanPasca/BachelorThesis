import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from datasets.paths import CAT_DATASET_SIGNAL_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH_64
from signal_analysis.signal_utils import get_filter_type, filter_input_sample
from utils.tf_utils import shuffle_indices

import matplotlib.pyplot as plt
import numpy as np


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

        self.cutoff_freq = cutoff_freq

        np.random.seed(random_seed)
        self.movies_to_keep = np.array(movies_to_keep)
        self._load_data(CAT_DATASET_SIGNAL_PATH,
                        CAT_DATASET_STIMULI_PATH_64, CAT_DATASET_STIMULI_PATH)
        self.number_of_channels = self.signal.shape[-2]
        self._normalize_data()
        self.classification_type = self._get_classification_type(model_output_type)
        self.nr_classes = self._get_nr_classes()
        self.model_output_type = model_output_type.upper()
        self.slicing_strategy = slicing_strategy.upper()
        self.split_by = split_by.upper()
        self.slice_length = slice_length
        self.nr_conditions = self.signal.shape[0]
        self.trials_per_condition = self.signal.shape[1]
        self._split_dataset(val_perc)

    def _split_dataset_by_trials(self, val_perc):
        self.train_indexes, self.val_indexes = shuffle_indices(
            self.trials_per_condition, val_perc, False)

        self.validation = self.signal[:, self.val_indexes, ...]
        self.train = self.signal[:, self.train_indexes, ...]

    def _split_dataset(self, val_perc):
        self.signal = self.signal[self.movies_to_keep, ...]

        if self.slicing_strategy == "CONSECUTIVE":
            nr_slices_per_channel = self.signal.shape[-1] // self.slice_length
            new_dataset = self.get_dataset_with_slices()

            if self.split_by == "TRIALS":
                train_indexes, val_indexes = shuffle_indices(
                    self.trials_per_condition, val_perc, False)
                self.train_slices = self.val_slices = np.arange(
                    0, nr_slices_per_channel)
                self.validation = new_dataset[:, val_indexes, ...]
                self.train = new_dataset[:, train_indexes, ...]

            elif self.split_by == "SLICES":
                self.train_slices, self.val_slices = shuffle_indices(
                    nr_slices_per_channel, val_perc, False)
                self.validation = new_dataset[:, :, :, self.val_slices, :]
                self.train = new_dataset[:, :, :, self.train_slices, :]

        elif self.slicing_strategy == "RANDOM":
            self._split_dataset_by_trials(val_perc)

    def get_dataset_with_slices(self):
        nr_slices_per_channel = self.signal.shape[-1] // self.slice_length
        new_dataset = np.zeros(
            (self.signal.shape[:-1] + (nr_slices_per_channel, self.slice_length)))
        for movie in range(self.signal.shape[0]):
            for trial in range(self.signal.shape[1]):
                slices = np.zeros((nr_slices_per_channel, self.slice_length))
                for channel in range(self.signal.shape[2]):
                    for i in range(nr_slices_per_channel):
                        slices[i] = self.signal[movie, trial, channel,
                                                i * self.slice_length: (i + 1) * self.slice_length]
                    new_dataset[movie, trial, channel] = slices

        return new_dataset

    def frame_generator(self, frame_size, batch_size, data, source):
        x = []
        y = []
        while 1:
            if self.slicing_strategy == "RANDOM":
                frame, image_causing_frame = self._get_random_frame_stimuli_trials(
                    frame_size, data)
            else:
                frame, image_causing_frame = self._get_random_frame_stimuli_slices(
                    data, source)

            x.append(frame.transpose())
            y.append(image_causing_frame)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.train, "train")

    def validation_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.validation, "val")

    def _get_random_frame_stimuli_trials(self, frame_size, data):
        random_sequence, (movie_index,
                          trial_index) = self.get_random_sequence(data)
        batch_start = np.random.choice(
            range(100, random_sequence.shape[-1] - frame_size))
        frame = random_sequence[:, batch_start:batch_start + frame_size]
        image_causing_frame = self._get_stimuli_for_sequence(
            movie_index, batch_start)

        return frame, image_causing_frame

    def _get_random_frame_stimuli_slices(self, data, source):
        random_sequence, (movie_index,
                          trial_index) = self.get_random_sequence(data)
        slice_index = np.random.randint(low=0, high=random_sequence.shape[1])
        frame = random_sequence[:, slice_index, :]
        if source.lower() == "val":
            timestamp = self.val_slices[slice_index] * self.slice_length
        else:
            timestamp = self.train_slices[slice_index] * self.slice_length
        image_causing_frame = self._get_stimuli_for_sequence(
            movie_index, timestamp)

        return frame, image_causing_frame

    def get_random_sequence(self, data_source):
        movie_index = np.random.choice(data_source.shape[0])
        trial_index = np.random.choice(data_source.shape[1])
        return data_source[movie_index, trial_index], (movie_index, trial_index)

    def _load_data(self, signal_path, stimuli_path_resized, stimuli_path):

        self.signal = np.load(signal_path)[self.movies_to_keep, ...]
        filter_type = get_filter_type(self.cutoff_freq)
        self.signal = filter_input_sample(
            self.signal, self.cutoff_freq, filter_type)
        self.stimuli = np.load(stimuli_path_resized)[self.movies_to_keep, ...]
        self.stimuli_mean = np.mean(
            self.stimuli / np.max(self.stimuli), axis=(2, 3))
        self.stimuli_orig_ = np.load(stimuli_path)[self.movies_to_keep, ...]
        self.stimuli_edges = np.zeros(self.stimuli.shape)
        for i, movie in enumerate(self.stimuli):
            for j, image in enumerate(movie):
                highThreshold = np.max(image) * 0.8
                lowThreshold = highThreshold * 0.9
                edges = cv.Canny(image, lowThreshold,
                                 highThreshold, L2gradient=True)
                kernel = np.ones((1, 1), np.uint8)
                edges = cv.morphologyEx(edges, cv.MORPH_OPEN, kernel)
                # kernel = np.ones((1, 2), np.uint8)
                # edges = cv.morphologyEx(edges, cv.MORPH_ERODE, kernel)
                self.stimuli_edges[i, j, :, :] = edges
        self.stimuli_edges_sum = np.mean(self.stimuli_edges, axis=(2, 3)) / 255

    def _get_stimuli_for_sequence(self, movie_index, seq_start):
        image_number = (seq_start - 100) // 40
        if self.model_output_type == "DCGAN":
            image = self.stimuli[movie_index, image_number, :, :]
            return image[:, :, np.newaxis]
        elif self.model_output_type == "BRIGHTNESS":
            return np.array([self.stimuli_mean[movie_index, image_number]])
        elif self.model_output_type == "EDGES":
            return np.array([self.stimuli_edges_sum[movie_index, image_number]])
        elif self.model_output_type == "CLASSIFY_MOVIES":
            return np.array([movie_index])

    def _normalize_data(self):
        for channel in range(self.signal.shape[2]):
            self.signal[:, :, channel,
                        :] /= np.max(self.signal[:, :, channel, :])
        self.stimuli = self.stimuli / np.max(self.stimuli)

    def _get_classification_type(self, classification_type):
        if classification_type.find("movies"):
            return "movies"
        elif classification_type.find("scenes"):
            return "scenes"

    def _get_nr_classes(self):
        if self.classification_type == "movies":
            return len(self.movies_to_keep)


if __name__ == '__main__':
    movies_to_keep = [0, 1, 2]
    dataset = CatLFPStimuli(
        val_perc=0.15, movies_to_keep=movies_to_keep, split_by="SLICES")
    np.random.seed(42)
    for movie in movies_to_keep:
        for j in np.random.choice(700, 10):
            plt.subplot(121), plt.imshow(
                dataset.stimuli[movie, j], cmap='gray')
            plt.title('Original Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(122), plt.imshow(
                dataset.stimuli_edges[movie, j], cmap='gray')
            plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
            plt.show()
