import csv
import json
import numpy as np
import os

from matplotlib import pyplot

from signal_utils import butter_lowpass_filter, rescale, mu_law_fn


class LFPDataset:

    def __init__(self, dataset_path, saved_as_npy=True, normalization=None, cutoff_freq=20, random_seed=42,
                 white_noise_dev=-1, nr_bins=256):
        self.description_file_path = dataset_path
        if saved_as_npy:
            self.load_from_npy(dataset_path)
        else:
            lfp_file_data = LFPDataset._parse_description(dataset_path)
            self.bin_file_names = lfp_file_data.get('bin_file_names')
            self.trial_length = lfp_file_data.get('trial_length')
            self.total_length = lfp_file_data.get('total_length')
            self.sampling_frequency = lfp_file_data.get('sampling_frequency')
            self.number_of_lfp_files = lfp_file_data.get('number_of_lfp_files')
            self.ldf_file_version = lfp_file_data.get('ldf_file_version')
            self.stimulus_condition_file = lfp_file_data.get('stimulus_condition_file')
            self.number_of_conditions = lfp_file_data.get('number_of_conditions')
            self.trials_per_condition = lfp_file_data.get('trials_per_condition')
            self.event_codes_file_path = lfp_file_data.get('event_codes_file')
            self.event_timestamps_file_path = lfp_file_data.get('event_timestamps_file')
            self.event_codes = self._parse_event_codes(lfp_file_data.get('event_codes_file'))
            self.event_timestamps = self._parse_event_timestamps(lfp_file_data.get('event_timestamps_file'))
            self.stimulus_conditions = self._parse_stimulus_data(lfp_file_data.get('stimulus_condition_file'))
            self.channels = self._parse_channels_data(lfp_file_data.get('bin_file_names'))
            self.stimulus_on_at, self.stimulus_off_at = self._parse_stimulus_on_off(lfp_file_data)

        self.random_seed = random_seed
        self.cutoff_freq = cutoff_freq
        self.nr_channels = len(self.channels)
        self.low_pass_filter = cutoff_freq > 0
        self.white_noise = white_noise_dev > 0
        self.white_noise_dev = white_noise_dev
        self.nr_bins = nr_bins

        if self.low_pass_filter:
            self._low_pass_input()

        self._normalize_input(normalization)

        if self.white_noise:
            self._add_noise_to_input()

    def _add_noise_to_input(self):
        noise = np.random.normal(0, self.white_noise_dev, self.channels.shape)
        self.channels += noise

    def _low_pass_input(self):
        for i, channel in enumerate(self.channels):
            self.channels[i] = butter_lowpass_filter(channel, self.cutoff_freq, 1000)

    def _normalize_input(self, normalization):
        self.mu_law = False
        if normalization is not None:
            if normalization == "Zsc":
                mean = np.mean(self.channels, axis=1, keepdims=True)
                std = np.std(self.channels, axis=1, keepdims=True)
                self.channels -= mean
                self.channels /= std

            elif normalization == "Brute":
                self.channels -= np.min(self.channels, axis=1, keepdims=True)
                self.channels /= np.max(self.channels, axis=1, keepdims=True)

            elif normalization == "MuLaw":
                """The signal is brought to [-1,1] through rescale->[-1,1] mu_law and then encoded using np.digitize"""
                self.limits = {}
                self.mu_law = True

                for i, channel in enumerate(self.channels):
                    np_min = np.min(channel)
                    np_max = np.max(channel)
                    self.limits[i] = (np_min, np_max)
                    self.channels[i] = mu_law_fn(
                        rescale(channel, old_max=np_max, old_min=np_min, new_max=1, new_min=-1), self.nr_bins)
                pyplot.hist(self.channels[15], 256)
                pyplot.show()

    def _parse_stimulus_data(self, condition_file_path):
        with open(os.path.join(os.path.dirname(self.description_file_path), condition_file_path),
                  'r') as f:
            if self.ldf_file_version == '1.0':
                return [int(st) for st in f.read().split('\n')]
            elif self.ldf_file_version == '1.1':
                csv_reader = csv.reader(f, delimiter=',')
                self.condition_refresh_rate = float(next(csv_reader)[1])
                self.experiment_duration = float(next(csv_reader)[1])
                col_descs = next(csv_reader)
                conditions = []
                for row in csv_reader:
                    cond = {}
                    for index, col in enumerate(col_descs):
                        cond[col] = row[index]
                    conditions.append(cond)
                return conditions

    def _parse_channels_data(self, channels_paths):
        return np.array(
            [np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), channel_path), 'rb'),
                         np.float32) for channel_path in channels_paths])

    def _parse_event_codes(self, file_path):
        if file_path is None:
            return None
        return np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), file_path), 'rb'),
                           np.int32)

    def _parse_event_timestamps(self, file_path):
        if file_path is None:
            return None
        return np.fromfile(open(os.path.join(os.path.dirname(self.description_file_path), file_path), 'rb'),
                           np.int32)

    def _parse_stimulus_on_off(self, lfp_description):
        if self.ldf_file_version == '1.0':
            return lfp_description.get('stimulus_on_at'), lfp_description.get('stimulus_off_at')
        elif self.ldf_file_version == '1.1':
            stimulus_on_at = []
            stimulus_off_at = []
            for index, event in enumerate(self.event_codes):
                if event == 129:
                    stimulus_on_at.append(self.event_timestamps[index])
                if event == 150:
                    stimulus_off_at.append(self.event_timestamps[index])
            return stimulus_on_at, stimulus_off_at

    def save_as_npy(self, path):
        np.save(path, vars(self))

    def load_from_npy(self, path):
        lfp_file_data = np.load(path).item()
        for prop, val in lfp_file_data.items():
            setattr(self, prop, val)

    @staticmethod
    def _parse_description(description_file_path):
        with open(description_file_path, 'r') as f:
            lfp_description = json.loads(f.read())
        return lfp_description

    def get_total_length(self, partition):
        if partition == "TRAIN":
            return self.train.size
        elif partition == "VAL":
            return self.validation.size
        elif partition == "TEST":
            return self.test.size
        else:
            raise ValueError("Please pick a valid partition from: TRAIN, VAL and TEST")

    def _compute_values_range(self, channels_to_keep=None):
        if channels_to_keep is None:
            min_val = np.min(self.channels)
            max_val = np.max(self.channels)
        else:
            min_val = np.min(self.channels[channels_to_keep])
            max_val = np.max(self.channels[channels_to_keep])
        self.values_range = min_val, max_val
