from datasets.LFPDataset import LFPDataset
from datasets.DATASET_PATHS import CAT_DATASET_PATH
import matplotlib.pyplot as plt
import numpy as np


class CatLFP(LFPDataset):
    def __init__(self, channels_to_keep=None, val_perc=0.20, test_perc=0.20, random_seed=42, nr_bins=256, nr_of_seqs=3):
        super().__init__(CAT_DATASET_PATH)
        self.nr_bins = nr_bins

        self._compute_values_range()
        self._pre_compute_bins()
        self._split_lfp_into_movies()
        if channels_to_keep is None:
            self.channels_to_keep = np.array(self.nr_channels)
        else:
            self.channels_to_keep = np.array(channels_to_keep)

        self._get_train_val_test_split_channel_wise(self.channels_to_keep, test_perc, val_perc)

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.prediction_sequences = {
            'val': [self.get_random_sequence_from('VAL') for _ in range(nr_of_seqs)],
            'train': [self.get_random_sequence_from('TRAIN') for _ in range(nr_of_seqs)]
        }

    def _split_lfp_into_movies(self):
        self.all_lfp_data = []
        for condition_id in range(0, self.number_of_conditions):
            condition = []
            for i in range(0, self.channels.shape[1] // self.trial_length):
                if self.stimulus_conditions[i] == condition_id + 1:
                    trial = self.channels[:, (i * self.trial_length):((i * self.trial_length) + self.trial_length)]
                    condition.append(trial)
            self.all_lfp_data.append(np.array(condition))
        self.all_lfp_data = np.array(self.all_lfp_data)
        self.channels = None

    def _pre_compute_bins(self):
        self.cached_val_bin = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]

    def _encode_input_to_bin(self, target_val):
        if target_val not in self.cached_val_bin:
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

    def altceva(self, val_perc, test_perc):
        channel_indexes = np.arange(self.nr_channels)
        np.random.shuffle(channel_indexes)
        nr_test_channels = round(test_perc * self.nr_channels)
        nr_val_channels = round(val_perc * self.nr_channels)
        nr_train_channels = self.nr_channels - nr_test_channels - nr_val_channels

        train_indexes = channel_indexes[:nr_train_channels]
        val_indexes = channel_indexes[nr_train_channels:nr_train_channels + nr_val_channels]
        test_indexes = channel_indexes[-nr_test_channels:]

        self.train = self.all_lfp_data[:, :, train_indexes, :]
        self.validation = self.all_lfp_data[:, :, val_indexes, :]
        self.test = self.all_lfp_data[:, :, test_indexes, :]

    def _get_train_val_test_split_channel_wise(self, channels_to_keep, val_perc, test_perc):
        nr_test_channels = round(test_perc * self.trials_per_condition)
        nr_val_channels = round(val_perc * self.trials_per_condition)
        nr_train_channels = self.trials_per_condition - nr_test_channels - nr_val_channels

        channel_indexes = np.arange(0, self.trials_per_condition)
        np.random.shuffle(channel_indexes)
        train_indexes = channel_indexes[:nr_train_channels]
        val_indexes = channel_indexes[nr_train_channels:nr_train_channels + nr_val_channels]
        test_indexes = channel_indexes[-nr_test_channels:]

        interm_data = self.all_lfp_data[:, :, channels_to_keep, :]

        self.train = interm_data[:, train_indexes, :].reshape(self.number_of_conditions, nr_train_channels, -1, 28000)
        self.validation = interm_data[:, val_indexes, :].reshape(self.number_of_conditions, nr_val_channels, -1,
                                                                 28000)
        self.test = interm_data[:, test_indexes, :].reshape(self.number_of_conditions, nr_test_channels, -1, 28000)

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

    def plot_signal(self, movie, trial, channel, start=0, stop=None, save=False, show=True):
        if stop is None:
            stop = self.trial_length
        plt.figure(figsize=(16, 12))
        plot_title = "Movie:{}_Channel:{}_Trial:{}_Start:{}_Stop:{}".format(movie, channel, trial, start, stop)
        plt.title(plot_title)
        signal = self.get_dataset_piece(movie, trial, channel)[start:stop]
        plt.plot(signal, label="LFP signal")
        plt.legend()
        if save:
            plt.savefig(
                "/home/pasca/School/Licenta/Datasets/CER01A50/Plots/" + "{}/".format(movie) + plot_title + ".png")
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
            Random sequence from the datasource n

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
