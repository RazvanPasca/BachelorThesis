import matplotlib.pyplot as plt
import numpy as np

from datasets.paths import CAT_DATASET_SIGNAL_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH
from signal_analysis.signal_utils import get_filter_type, filter_input_sample


class CatLFPStimuli:
    def __init__(self, movies_to_keep=[0, 1, 2], cutoff_freq=None, val_perc=0.20, random_seed=42):

        self.cutoff_freq = cutoff_freq

        np.random.seed(random_seed)
        self.movies_to_keep = np.array(movies_to_keep)
        self._load_data(CAT_DATASET_SIGNAL_PATH, CAT_DATASET_STIMULI_PATH)
        self.number_of_channels = self.signal.shape[-2]
        self._normalize_data()
        self.nr_conditions = self.signal.shape[0]
        self.trials_per_condition = self.signal.shape[1]
        self._split_dataset(val_perc)

    def _retrieve_trials(self, indexes):
        movies = []
        for i in range(self.signal.shape[0]):
            movies.append(self.signal[i, indexes, :, :])
        return np.array(movies)

    def _split_dataset(self, val_perc):
        nr_val_trials = round(val_perc * self.trials_per_condition)
        nr_train_trials = self.trials_per_condition - nr_val_trials

        trial_indexes_shuffled = np.arange(0, self.trials_per_condition)
        np.random.shuffle(trial_indexes_shuffled)
        self.train_indexes = trial_indexes_shuffled[:nr_train_trials]
        self.val_indexes = trial_indexes_shuffled[nr_train_trials:nr_train_trials + nr_val_trials]

        self.validation = self._retrieve_trials(self.val_indexes)
        self.train = self._retrieve_trials(self.train_indexes)

    def frame_generator(self, frame_size, batch_size, data, data_indexes):
        x = []
        y = []
        while 1:
            frame, image_causing_frame = self._get_random_frame_stimuli(frame_size, data, data_indexes)
            x.append(frame.transpose())
            y.append(image_causing_frame[:, :, np.newaxis])
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.train, self.train_indexes)

    def validation_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.validation, self.val_indexes)

    def _get_random_frame_stimuli(self, frame_size, data, data_indexes):
        random_sequence, (movie_index, trial_index) = self.get_random_sequence(data)
        batch_start = np.random.choice(range(100, random_sequence.shape[-1] - frame_size))
        frame = random_sequence[:, batch_start:batch_start + frame_size]
        image_causing_frame = self._get_stimuli_for_sequence(movie_index, batch_start, batch_start + frame_size)

        return frame, image_causing_frame

    def get_random_sequence(self, data_source):
        movie_index = np.random.choice(data_source.shape[0])
        trial_index = np.random.choice(data_source.shape[1])
        return data_source[movie_index, trial_index], (movie_index, trial_index)

    def _load_data(self, signal_path, stimuli_path, ):

        self.signal = np.load(signal_path)[self.movies_to_keep, ...]
        filter_type = get_filter_type(self.cutoff_freq)
        self.signal = filter_input_sample(self.signal, self.cutoff_freq, filter_type)
        self.stimuli = np.load(stimuli_path)[self.movies_to_keep, ...]

    def _get_stimuli_for_sequence(self, movie_index, seq_start, seq_end):
        image_number = (seq_start - 100) // 40
        return self.stimuli[movie_index, image_number, :, :]

    def _normalize_data(self):
        for channel in range(self.signal.shape[2]):
            self.signal[:, :, channel, :] /= np.max(self.signal[:, :, channel, :])
        self.stimuli = self.stimuli / np.max(self.stimuli)


if __name__ == '__main__':
    dataset = CatLFPStimuli(movies_to_keep=[0], val_perc=0.15)
    for x, y in dataset.train_frame_generator(100, 2):
        print(x.shape)
        print(y.shape)
        plt.imshow(y[0].reshape((64, 64)))
        plt.show()
