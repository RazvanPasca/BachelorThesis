import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot

from datasets.LFPDataset import LFPDataset


class MouseLFP(LFPDataset):
    def __init__(self, dataset_path, frame_size=512, channels_to_keep=None, conditions_to_keep=None, val_perc=0.20,
                 test_perc=0.0,
                 random_seed=42, nr_bins=256,
                 nr_of_seqs=3,
                 normalization="Zsc",
                 cutoff_freq=50):
        super().__init__(dataset_path, normalization=normalization, cutoff_freq=cutoff_freq)
        np.random.seed(random_seed)

        self.frame_size = frame_size
        self.nr_bins = nr_bins
        self.random_seed = random_seed
        self.normalization = normalization
        self.nr_channels = len(self.channels)
        self.nr_of_orientations = 8
        self.nr_of_stimulus_luminosity_levels = 3
        self.number_of_conditions = 24
        self.trial_length = 2672  # 4175

        if channels_to_keep is None:
            self.channels_to_keep = np.array(range(self.nr_channels))
        else:
            self.channels_to_keep = np.array(channels_to_keep)

        self._compute_values_range(channels_to_keep=[15])
        self._pre_compute_bins()
        self._split_lfp_data()

        if conditions_to_keep is None:
            self.conditions_to_keep = np.array(range(self.number_of_conditions))
        else:
            self.conditions_to_keep = np.array(conditions_to_keep) - 1
            self.number_of_conditions = len(conditions_to_keep)

        self._get_train_val_test_split_channel_wise(self.channels_to_keep, self.conditions_to_keep, val_perc, test_perc)

        self.prediction_sequences = {
            'val': [self.get_random_sequence_from('VAL') for _ in range(nr_of_seqs)],
            'train': [self.get_random_sequence_from('TRAIN') for _ in range(nr_of_seqs)]
        }

        np.random.seed(datetime.datetime.now().microsecond)

    def _split_lfp_data(self):
        self.all_lfp_data = []
        for condition in range(1, self.number_of_conditions + 1):
            conditions = []
            for stimulus_condition in self.stimulus_conditions:
                if stimulus_condition['Condition number'] == str(condition):
                    index = int(stimulus_condition['Trial']) - 1
                    events = [{'timestamp': self.event_timestamps[4 * index + i],
                               'code': self.event_codes[4 * index + i]} for i in range(4)]
                    trial = self.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)]
                    # Right now it cuts only the area where the stimulus is active
                    # In order to keep the whole trial replace with
                    # "trial = self.channels[:, events[0]['timestamp']:(events[0]['timestamp'] + 4175)]"
                    conditions.append(trial)
            self.all_lfp_data.append(np.array(conditions))
        self.all_lfp_data = np.array(self.all_lfp_data, dtype=np.float32)
        self.channels = None

    def _pre_compute_bins(self):
        self.cached_val_bin = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        pyplot.hist(self.channels[15], self.nr_bins)
        pyplot.title(
            "After applying norm with Lpass{} and {}".format(self.cutoff_freq, self.normalization))
        pyplot.show()

        self.bin_size = self.bins[1] - self.bins[0]

    def _encode_input_to_bin(self, target_val):
        if target_val not in self.cached_val_bin:
            if self.mu_law:
                self.cached_val_bin[target_val] = self.mu_law_encoding(target_val)
            else:
                self.cached_val_bin[target_val] = np.digitize(target_val, self.bins, right=False)
        return self.cached_val_bin[target_val]

    def _get_train_val_test_split_channel_wise(self, channels_to_keep, conditions_to_keep, val_perc, test_perc):
        nr_test_trials = round(test_perc * self.trials_per_condition)
        nr_val_trials = round(val_perc * self.trials_per_condition)
        nr_train_trials = self.trials_per_condition - nr_test_trials - nr_val_trials

        trial_indexes = np.arange(0, self.trials_per_condition)
        np.random.shuffle(trial_indexes)
        train_indexes = trial_indexes[:nr_train_trials]
        val_indexes = trial_indexes[nr_train_trials:nr_train_trials + nr_val_trials]
        test_indexes = trial_indexes[-nr_test_trials:]

        interm_data = self.all_lfp_data[conditions_to_keep, :, channels_to_keep, :]

        self.train = interm_data[:, train_indexes, :].reshape(self.number_of_conditions, nr_train_trials, -1,
                                                              self.trial_length)
        self.validation = interm_data[:, val_indexes, :].reshape(self.number_of_conditions, nr_val_trials, -1,
                                                                 self.trial_length)
        if nr_test_trials <= 0:
            pass
        else:
            self.test = interm_data[:, test_indexes, :].reshape(self.number_of_conditions, nr_test_trials, -1,
                                                                self.trial_length)

    def _get_train_val_test_split_with_indexes(self, channels_to_keep, conditions_to_keep, val_perc, test_perc,
                                               indexes):
        self._get_train_val_test_split_channel_wise(channels_to_keep, conditions_to_keep, val_perc, test_perc)
        train_slices = []
        val_slices = []

        train_shape = self.train.shape
        val_shape = self.validation.shape
        slice_length = indexes[0][1] - indexes[0][0]

        for index_pair in indexes:
            train_slices.append(self.train[:, :, :, index_pair[0]:index_pair[1]])
            val_slices.append(self.validation[:, :, :, index_pair[0]:index_pair[1]])

        train_slices = np.array(train_slices)
        val_slices = np.array(val_slices)
        self.train = train_slices.reshape(train_shape[0], train_shape[1], -1, slice_length)
        self.validation = val_slices.reshape(val_shape[0], val_shape[1], -1, slice_length)

        assert (np.all(self.train[0, 0,]))

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

    def get_dataset_piece(self, condition, trial, channel):
        return self.all_lfp_data[condition, trial, channel, :]

    def plot_signal(self, condition, trial, channel, start=0, stop=None, save_path=None, show=True):
        if stop is None:
            stop = self.trial_length
        plt.figure(figsize=(16, 12))
        plot_title = "Condition:{}_Channel:{}_Trial:{}_Start:{}_Stop:{}".format(condition, channel, trial, start, stop)
        plt.title(plot_title)
        signal = self.get_dataset_piece(condition, trial, channel)[start:stop]
        plt.plot(signal, label="Mouse LFP signal")
        plt.legend()
        if save_path is not None:
            plt.savefig(os.path.join(save_path, "Mouse/Plots/{0}.png".format(str(condition) + plot_title)))
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
        condition_index = np.random.choice(data_source.shape[0])
        trial_index = np.random.choice(data_source.shape[1])
        channel_index = np.random.choice(data_source.shape[2])
        return self.get_sequence(condition_index, trial_index, channel_index, data_source)

    def get_sequence(self, condition_index, trial_index, channel_index, data_source):
        data_address = {
            'Condition': condition_index,
            'T': trial_index,
            'C': channel_index
        }
        return data_source[data_address['Condition'], data_address['T'], data_address['C'], :], data_address
