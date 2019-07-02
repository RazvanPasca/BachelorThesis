import os
import matplotlib.pyplot as plt
import numpy as np
from utils.tf_utils import replace_at_index


class MouseLFP():
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
                 noisy_channels=(3, 6, 32),
                 gamma_windows_in_trial=None,
                 condition_on_gamma=False):
        super().__init__(dataset_path,
                         normalization=normalization,
                         cutoff_freq=cutoff_freq,
                         random_seed=random_seed,
                         white_noise_dev=white_noise_dev,
                         nr_bins=nr_bins)

        np.random.seed(random_seed)
        self.frame_size = frame_size
        self.normalization = normalization
        self.nr_channels = len(self.channels)
        self.nr_of_orientations = 8
        self.nr_of_stimulus_luminosity_levels = 3
        self.number_of_conditions = 24
        self.trial_length = 2672  # 2672  # 4175
        self.nr_of_seqs = nr_of_seqs
        self.condition_on_gamma = condition_on_gamma
        self.gamma_info = self.compute_gamma_labels_for_trial(gamma_windows_in_trial)
        self.gamma_windows_in_trial = [item for sublist in gamma_windows_in_trial for item in sublist if
                                       item < self.trial_length]

        self._get_dataset_keep_indexes(channels_to_keep, conditions_to_keep, trials_to_keep, noisy_channels, )
        self.get_sequences_for_plotting()


    def get_sequences_for_plotting(self):
        self.benchmark_sequences = {
            'VAL': [],
            'TRAIN': []
        }
        nr_signals_in_val = min(6, self.validation.shape[2] * self.validation.shape[1] * self.validation.shape[0])

        seq_nr = 0
        for channel_nr in range(len(self.channels_to_keep)):
            for cond_index in range(len(self.conditions_to_keep)):
                for trial_index in range(len(self.trials_to_keep)):
                    channel_nr = channel_nr % self.validation.shape[2]
                    if seq_nr < nr_signals_in_val:
                        seq_nr += 1
                        for key in self.benchmark_sequences.keys():
                            sequence_and_source = self.get_sequence_from(cond_index, trial_index, channel_nr, key)
                            gamma_label = self.gamma_info
                            sequence_w_gamma_labels = np.concatenate(
                                (sequence_and_source[0].reshape(self.trial_length, 1), gamma_label), axis=1)
                            sequence_and_source = replace_at_index(sequence_and_source, 0, sequence_w_gamma_labels)
                            self.benchmark_sequences[key].append(sequence_and_source)
                    else:
                        break

    def compute_gamma_labels_for_trial(self, gamma_windows_in_trial):
        trial_gamma_label = np.zeros(self.trial_length)
        if self.condition_on_gamma:
            gamma_ranges = [list(range(l[0], l[1] + 1)) for l in gamma_windows_in_trial]
            gamma_ranges = [item for sublist in gamma_ranges for item in sublist if item < self.trial_length]
            np.put(trial_gamma_label, gamma_ranges, 1)
        return trial_gamma_label.reshape(-1, 1)

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
                gamma_label = self.gamma_info[batch_start:batch_start + frame_size]
                input = np.concatenate((frame.reshape(frame_size, 1), gamma_label), axis=1)
                next_step_value = random_sequences[elem][batch_start + frame_size]
                x.append(input)
                y.append(next_step_value)

            if len(x) == batch_size:
                y = self._get_y_value(classifying, y)

                yield np.array(x), y.reshape(batch_size, 1)
                x = []
                y = []

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
