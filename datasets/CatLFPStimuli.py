import matplotlib.pyplot as plt
import numpy as np

from datasets.paths import CAT_DATASET_SIGNAL_PATH
from datasets.paths import CAT_DATASET_STIMULI_PATH


class CatLFPStimuli:
    def __init__(self,
                 val_perc=0.20,
                 test_perc=0.0,
                 random_seed=42):

        np.random.seed(random_seed)

        self._load_data(CAT_DATASET_SIGNAL_PATH, CAT_DATASET_STIMULI_PATH)
        self._normalize_data()
        self.nr_conditions = self.signal.shape[0]
        self.trials_per_condition = self.signal.shape[1]
        self._split_dataset(val_perc, test_perc)


    def _split_dataset(self, val_perc, test_perc):
        nr_of_trials = self.nr_conditions * self.trials_per_condition
        nr_test_trials = round(test_perc * nr_of_trials)
        nr_val_trials = round(val_perc * nr_of_trials)
        nr_train_trials = nr_of_trials - (nr_test_trials + nr_val_trials)

        trial_indexes_shuffled = np.arange(0, nr_of_trials)
        np.random.shuffle(trial_indexes_shuffled)
        self.train_indexes = trial_indexes_shuffled[:nr_train_trials]
        self.val_indexes = trial_indexes_shuffled[nr_train_trials:nr_train_trials + nr_val_trials]
        self.test_indexes = trial_indexes_shuffled[-nr_test_trials:]

        temp_data = self.signal.reshape((60, 47, 28000))
        self.validation = temp_data[self.val_indexes, :]
        self.test = temp_data[self.test_indexes, :]
        self.train = temp_data[self.train_indexes, :]

    def frame_generator(self, frame_size, batch_size, data, data_indexes):
        x = []
        y = []
        while 1:
            random_sequence, trial_index = self.get_random_sequence(data)
            batch_start = np.random.choice(range(100, random_sequence.shape[-1] - frame_size - 100))
            frame = random_sequence[:, batch_start:batch_start + frame_size]
            image_causing_frame = self._get_stimuli_for_sequence(trial_index, data_indexes, batch_start,
                                                                 batch_start + frame_size)
            x.append(frame.reshape(frame_size, 47))
            y.append(image_causing_frame[:, :, np.newaxis])
            if len(x) == batch_size:
                yield np.array(x), np.array(y)
                x = []
                y = []

    def train_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.train, self.train_indexes)

    def validation_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.validation, self.val_indexes)

    def test_frame_generator(self, frame_size, batch_size):
        return self.frame_generator(frame_size, batch_size, self.test, self.test_indexes)

    def get_random_sequence(self, data_source):
        trial_index = np.random.choice(data_source.shape[0])
        return data_source[trial_index], trial_index

    def _load_data(self, signal_path, stimuli_path):
        self.signal = np.load(signal_path)
        self.stimuli = np.load(stimuli_path)

    def _get_stimuli_for_sequence(self, trial_index, data_indexes, seq_start, seq_end):
        index_in_all_trials = data_indexes[trial_index]
        condition = index_in_all_trials // self.trials_per_condition
        image_number = (seq_start - 100) // 40
        return self.stimuli[condition, image_number, :, :]

    def _normalize_data(self):
        for channel in range(self.signal.shape[2]):
            self.signal[:, :, channel, :] /= np.max(self.signal[:, :, channel, :])
        self.stimuli = self.stimuli / np.max(self.stimuli)


if __name__ == '__main__':
    dataset = CatLFPStimuli()
    for x, y in dataset.train_frame_generator(100, 2):
        print(x.shape)
        print(y.shape)
        plt.imshow(y[0])
        plt.show()
