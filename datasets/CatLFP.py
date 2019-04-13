import datetime
import os

from datasets.LFPDataset import LFPDataset
from datasets.DATASET_PATHS import PASCA_CAT_DATASET_PATH as CAT_DATASET_PATH
import matplotlib.pyplot as plt
import numpy as np


class CatLFP(LFPDataset):
    def __init__(self, movies_to_keep=None, channels_to_keep=None, val_perc=0.20, test_perc=0.0, random_seed=42,
                 nr_bins=256, nr_of_seqs=6, normalization="Zsc", cutoff_freq=50):
        super().__init__(CAT_DATASET_PATH, normalization=normalization, cutoff_freq=cutoff_freq,
                         random_seed=random_seed)

        np.random.seed(random_seed)
        self.nr_bins = nr_bins
        self.normalization = normalization
        self._compute_values_range()
        self._pre_compute_bins()
        self._split_lfp_into_movies()

        if channels_to_keep is None:
            self.channels_to_keep = np.array(range(self.nr_channels))
        else:
            self.channels_to_keep = np.array(channels_to_keep)

        if movies_to_keep is None:
            self.movies_to_keep = np.array(range(self.number_of_conditions))
        else:
            self.movies_to_keep = np.array(movies_to_keep)

        self._get_train_val_test_split_channel_wise(self.movies_to_keep, self.channels_to_keep, val_perc, test_perc)
        self.trial_length = self.train.shape[-1]

        self.prediction_sequences = {
            'val': [self.get_random_sequence_from('VAL') for _ in range(nr_of_seqs)],
            'train': [self.get_random_sequence_from('TRAIN') for _ in range(nr_of_seqs)]
        }

        np.random.seed(datetime.datetime.now().microsecond)

    def _split_lfp_into_movies(self):
        self.all_lfp_data = []
        for condition_id in range(0, self.number_of_conditions):
            condition = []
            for i in range(0, self.channels.shape[1] // self.trial_length):
                if self.stimulus_conditions[i] == condition_id + 1:
                    trial = self.channels[:, (i * self.trial_length):((i * self.trial_length) + self.trial_length)]
                    condition.append(trial)
            self.all_lfp_data.append(np.array(condition))

        self.all_lfp_data = np.array(self.all_lfp_data, dtype=np.float32)
        self.channels = None

    def _pre_compute_bins(self):
        self.cached_val_bin = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]

    def _encode_input_to_bin(self, target_val):
        if target_val not in self.cached_val_bin:
            if self.mu_law:
                self.cached_val_bin[target_val] = self.mu_law_encoding(target_val)
            else:
                self.cached_val_bin[target_val] = np.digitize(target_val, self.bins, right=False)
        return self.cached_val_bin[target_val]

    def _get_train_val_test_split(self, val_perc, test_perc, random=False):
        self.val_length = round(val_perc * self.trial_length)
        self.test_length = round(test_perc * self.trial_length)
        self.train_length = self.trial_length - (self.val_length + self.test_length)
        if not random:
            self.train = self.all_lfp_data[:, :, :, :self.train_length]
            self.validation = self.all_lfp_data[:, :, :, self.train_length:self.train_length + self.val_length]
            self.test = self.all_lfp_data[:, :, :,
                        self.train_length + self.val_length:self.train_length + self.val_length + self.test_length]

    def _get_train_val_test_split_channel_wise(self, movies_to_keep, channels_to_keep, val_perc, test_perc):
        nr_test_trials = round(test_perc * self.trials_per_condition)
        nr_val_trials = round(val_perc * self.trials_per_condition)
        nr_train_trials = self.trials_per_condition - nr_test_trials - nr_val_trials

        trial_indexes_shuffled = np.arange(0, self.trials_per_condition)
        np.random.shuffle(trial_indexes_shuffled)
        train_indexes = trial_indexes_shuffled[:nr_train_trials]
        val_indexes = trial_indexes_shuffled[nr_train_trials:nr_train_trials + nr_val_trials]
        test_indexes = trial_indexes_shuffled[-nr_test_trials:]

        interm_data = self.all_lfp_data[movies_to_keep, :, channels_to_keep, :]

        self.train = interm_data[:, train_indexes, :].reshape(movies_to_keep.size, nr_train_trials, -1, 28000)

        self.validation = interm_data[:, val_indexes, :].reshape(movies_to_keep.size, nr_val_trials, -1, 28000)

        assert (
            np.all(self.train[0, 0, 0] == self.all_lfp_data[movies_to_keep[0], train_indexes[0], channels_to_keep[0]]))

        assert (np.all(self.train[movies_to_keep.size - 1, train_indexes.size - 1, channels_to_keep.size - 1] ==
                       self.all_lfp_data[
                           movies_to_keep[-1], train_indexes[-1], channels_to_keep[-1]]))

        assert (np.all(self.validation[0, 0, 0] == self.all_lfp_data[
            movies_to_keep[0], val_indexes[0], channels_to_keep[0]]))

        assert (np.all(self.validation[movies_to_keep.size - 1, val_indexes.size - 1, channels_to_keep.size - 1] ==
                       self.all_lfp_data[
                           movies_to_keep[-1], val_indexes[-1], channels_to_keep[-1]]))

        if nr_test_trials > 0:
            self.test = interm_data[:, test_indexes, :].reshape(movies_to_keep.size, nr_test_trials, -1, 28000)

    def frame_generator(self, frame_size, batch_size, classifying, data):
        x = []
        y = []
        while 1:
            random_sequence, _ = self.get_random_sequence(data)
            batch_start = np.random.choice(range(0, random_sequence.size - frame_size))
            frame = random_sequence[batch_start:batch_start + frame_size]
            next_step_value = random_sequence[batch_start + frame_size]
            x.append(frame.reshape(frame_size, 1))
            y.append(self._encode_input_to_bin(next_step_value) if classifying else next_step_value)
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.train)

    def validation_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.validation)

    def test_frame_generator(self, frame_size, batch_size, classifying):
        return self.frame_generator(frame_size, batch_size, classifying, self.test)

    def get_dataset_piece(self, movie, trial, channel):
        return self.all_lfp_data[movie, trial, channel, :]

    def plot_signal(self, movie, trial, channel, start=0, stop=None, save_path=None, show=True):
        if stop is None:
            stop = self.trial_length
        plt.figure(figsize=(16, 12))
        plot_title = "Movie:{}_Channel:{}_Trial:{}_Start:{}_Stop:{}".format(movie, channel, trial, start, stop)
        plt.title(plot_title)
        signal = self.get_dataset_piece(movie, trial, channel)[start:stop]
        plt.plot(signal, label="LFP signal")
        plt.legend()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "Cat/Plots/{}/{}.png".format(movie, plot_title)))
        if show:
            plt.show()

    def get_random_sequence_from(self, source='VAL'):
        if source == 'TRAIN':
            data_source = self.train
        elif source == 'VAL':
            data_source = self.validation
        elif source == 'TEST':
            data_source = self.test
        else:
            raise ValueError("Please pick one out of TRAIN, VAL, TEST as data source")

        sequence, sequence_addr = self.get_random_sequence(data_source)
        sequence_addr['SOURCE'] = source
        return sequence, sequence_addr

    def get_random_sequence(self, data_source):
        """Function which receives data source as parameter and chooses randomly and index
        from each of the first 3 dimensions and returns the corresponding sequence
        and a dictionary containing the source indexes

        Args:
            data_source:a 4D numpy array: (movies, trials, channels, nr_samples)
                A random index is chosen from each of the first
                3 dimensions to pick the random sequence

        Returns:
            Random sequence from the datasource

             Dictionary with the format of
             'M': movie_index,
             'T': trial_index,
             'C': channel_index
        """
        movie_index = np.random.choice(data_source.shape[0])
        trial_index = np.random.choice(data_source.shape[1])
        channel_index = np.random.choice(data_source.shape[2])
        return self.get_sequence(movie_index, trial_index, channel_index, data_source)

    def get_sequence(self, movie_index, trial_index, channel_index, data_source):
        data_address = {
            'M': movie_index,
            'T': trial_index,
            'C': channel_index
        }
        return data_source[data_address['M'], data_address['T'], data_address['C'], :], data_address


if __name__ == '__main__':
    dataset = CatLFP()
    print(dataset.all_lfp_data.shape)
