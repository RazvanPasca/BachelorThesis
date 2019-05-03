import datetime
import os

import matplotlib.pyplot as plt
import numpy as np

from datasets.LFPDataset import LFPDataset
from signal_utils import mu_law_encoding, mu_law_fn, rescale


class MouseLFP(LFPDataset):
    def __init__(self, dataset_path, frame_size=512,
                 channels_to_keep=(-1,),
                 conditions_to_keep=(-1,),
                 trials_to_keep=(-1,),
                 val_perc=0.20,
                 random_seed=42,
                 nr_bins=256,
                 nr_of_seqs=6,
                 normalization="Zsc",
                 cutoff_freq=50,
                 white_noise_dev=-1,
                 noisy_channels=(3, 6, 30, 22, 23, 19, 18, 32)):
        super().__init__(dataset_path, normalization=normalization, cutoff_freq=cutoff_freq, random_seed=random_seed,
                         white_noise_dev=white_noise_dev, nr_bins=nr_bins)

        np.random.seed(random_seed)
        self.frame_size = frame_size
        self.normalization = normalization
        self.nr_channels = len(self.channels)
        self.nr_of_orientations = 8
        self.nr_of_stimulus_luminosity_levels = 3
        self.number_of_conditions = 24
        self.trial_length = 2672  # 2672  # 4175
        self.nr_of_seqs = nr_of_seqs

        self._split_lfp_data()

        self._get_dataset_keep_indexes(channels_to_keep, conditions_to_keep, trials_to_keep, noisy_channels, )
        self.get_train_val_split(val_perc)

        self.all_lfp_data = self._normalize_input_per_channel(self.all_lfp_data)

        self._pre_compute_bins()
        self.get_sequences_for_plotting()

        np.random.seed(datetime.datetime.now().microsecond)

    def get_sequences_for_plotting(self):
        self.prediction_sequences = {
            'VAL': [],
            'TRAIN': []
        }
        for seq_nr in range(self.validation.shape[2]):
            cond_index = np.random.choice(len(self.conditions_to_keep))
            trial_index = np.random.choice(len(self.trials_to_keep))
            for key in self.prediction_sequences.keys():
                self.prediction_sequences[key].append(self.get_sequence_from(cond_index, trial_index, seq_nr, key))

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
                    # trial = self.channels[:, events[0]['timestamp']:(events[0]['timestamp'] + 4175)]

                    conditions.append(trial)
            self.all_lfp_data.append(np.array(conditions))
        self.all_lfp_data = np.array(self.all_lfp_data, dtype=np.float32)
        self.channels = None

    def _normalize_input_per_channel(self, input_data):
        self.mu_law = False

        nr_channels = input_data.shape[2]

        if self.normalization == "Zsc":
            for i in range(nr_channels):
                mean = np.mean(input_data[:, :, i, :], keepdims=True)
                std = np.std(input_data[:, :, i, :], keepdims=True)
                input_data[:, :, i, :] -= mean
                input_data[:, :, i, :] /= std

        elif self.normalization == "Brute":
            for i in range(nr_channels):
                min = np.min(input_data[:, :, i, :], keepdims=True)
                max = np.max(input_data[:, :, i, :], keepdims=True)
                input_data[:, :, i, :] -= min
                input_data[:, :, i, :] /= max

        elif self.normalization == "MuLaw":
            """The signal is brought to [-1,1] through rescale->[-1,1] mu_law and then encoded using np.digitize"""
            self.limits = {}
            self.mu_law = True

            for i in range(nr_channels):
                np_min = np.min(input_data[:, :, i, :], keepdims=True)
                np_max = np.max(input_data[:, :, i, :], keepdims=True)
                self.limits[i] = (np_min, np_max)
                input_data[:, :, i, :] = mu_law_fn(
                    rescale(input_data[:, :, i, :], old_max=np_max, old_min=np_min, new_max=1, new_min=-1),
                    self.nr_bins)

        return input_data

    def _pre_compute_bins(self):
        self._compute_values_range()
        self.cached_val_bin = {}
        min_train_seq = np.floor(self.values_range[0])
        max_train_seq = np.ceil(self.values_range[1])
        self.bins = np.linspace(min_train_seq, max_train_seq, self.nr_bins)
        self.bin_size = self.bins[1] - self.bins[0]

    def _encode_input_to_bin(self, target_val):
        if target_val not in self.cached_val_bin:
            if self.mu_law:
                self.cached_val_bin[target_val] = mu_law_encoding(target_val, self.nr_bins)
            else:
                self.cached_val_bin[target_val] = np.digitize(target_val, self.bins, right=False)
        return self.cached_val_bin[target_val]

    def get_train_val_split(self, val_perc):
        nr_val_series = round(val_perc * self.channels_to_keep.size)
        nr_train_series = self.channels_to_keep.size - nr_val_series
        channels_shuffled_indexes = self.channels_to_keep
        np.random.shuffle(channels_shuffled_indexes)

        train_indexes = channels_shuffled_indexes[:nr_train_series]
        val_indexes = channels_shuffled_indexes[nr_train_series:]

        interm_data = self.all_lfp_data[self.conditions_to_keep]

        interm_data = interm_data[:, self.trials_to_keep, ...]

        p_interm_data = self._normalize_input_per_channel(interm_data[:, :, channels_shuffled_indexes, :])

        self.train = p_interm_data[:, :, :nr_train_series, :]
        self.validation = p_interm_data[:, :, nr_train_series:, :]

        self.channels_lookup = {"VAL": {},
                                "TRAIN": {}}

        for i, channel in enumerate(train_indexes):
            self.channels_lookup["TRAIN"][i] = channel

        for i, channel in enumerate(val_indexes):
            self.channels_lookup["VAL"][i] = channel

    def frame_generator(self, frame_size, batch_size, classifying, data):
        x = []
        y = []
        while 1:
            random_sequences = [self.get_random_sequence(data)[0] for _ in range(batch_size)]
            batch_starts = np.random.choice(range(0, random_sequences[0].size - frame_size), batch_size)

            for elem in range(batch_size):
                batch_start = batch_starts[elem]
                frame = random_sequences[elem][batch_start:batch_start + frame_size]
                next_step_value = random_sequences[elem][batch_start + frame_size]
                x.append(frame.reshape(frame_size, 1))
                y.append(next_step_value)

            if len(x) == batch_size:
                y = self._get_y_value(classifying, y)

                yield np.array(x), y
                x = []
                y = []

    def _get_y_value(self, classifying, y):
        """Classifying codes: 2 is MSE_CE, 1 is CE , -1 is Regression"""
        if classifying == 2:
            y = {'Regression': np.array(y),
                 "Sfmax": np.array([self._encode_input_to_bin(x) for x in y])}
        elif classifying == 1:
            y = np.array([self._encode_input_to_bin(x) for x in y])
        else:
            y = np.array(y)
        return y

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
        sequence_addr['C'] = self.channels_lookup[source][sequence_addr['C']]
        return sequence, sequence_addr

    def get_sequence_from(self, condition_index, trial_index, channel_index, source='VAL'):
        if source == 'TRAIN':
            data_source = self.train
        elif source == 'VAL':
            data_source = self.validation
        elif source == 'TEST':
            data_source = self.test
        else:
            raise ValueError("Please pick one out of TRAIN, VAL, TEST as data source")

        sequence, sequence_addr = self.get_sequence(condition_index, trial_index, channel_index, data_source)
        sequence_addr['SOURCE'] = source
        sequence_addr['C'] = self.channels_lookup[source][sequence_addr['C']]
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
        return data_source[data_address['Condition'], data_address['T'], channel_index, :], data_address

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

        self.channels_lookup = {}
        for i, channel in enumerate(channels_to_keep):
            self.channels_lookup[i] = channel

        self.train = interm_data[:, train_indexes, :].reshape(self.number_of_conditions, nr_train_trials, -1,
                                                              self.trial_length)
        self.validation = interm_data[:, val_indexes, :].reshape(self.number_of_conditions, nr_val_trials, -1,
                                                                 self.trial_length)
        if nr_test_trials <= 0:
            pass
        else:
            self.test = interm_data[:, test_indexes, :].reshape(self.number_of_conditions, nr_test_trials, -1,
                                                                self.trial_length)
