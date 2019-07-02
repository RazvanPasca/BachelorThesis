import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from datasets.oldies.LFPDataset import LFPDataset

from datasets.paths import CAT_DATASET_PATH_50


class CatLFP(LFPDataset):
    def __init__(self, conditions_to_keep=(-1,),
                 channels_to_keep=(-1,),
                 trials_to_keep=(-1,),
                 noisy_channels=(),
                 val_perc=0.20,
                 test_perc=0.0,
                 random_seed=42,
                 nr_bins=256,
                 nr_of_seqs=6,
                 normalization="Zsc",
                 cutoff_freq=50,
                 white_noise_dev=-1):
        super().__init__(CAT_DATASET_PATH_50, normalization=normalization, cutoff_freq=cutoff_freq,
                         random_seed=random_seed, white_noise_dev=white_noise_dev, nr_bins=nr_bins)

        np.random.seed(random_seed)
        self.normalization = normalization

        self._split_lfp_data()

        self._get_dataset_keep_indexes(channels_to_keep, conditions_to_keep, trials_to_keep, noisy_channels, )
        self._get_train_val_test_split_channel_wise(self.conditions_to_keep, self.channels_to_keep, val_perc, test_perc)

        self._pre_compute_bins()
        self.get_sequences_for_plotting(nr_of_seqs)

        np.random.seed(datetime.datetime.now().microsecond)

    """
    Decides which sequences are used for plotting
    """
    def get_sequences_for_plotting(self, nr_of_seqs):
        self.prediction_sequences = {
            'VAL': [self.get_random_sequence_from('VAL') for _ in range(nr_of_seqs)],
            'TRAIN': [self.get_random_sequence_from('TRAIN') for _ in range(nr_of_seqs)]
        }

    """
    Gets a full channels sequence using movie trial and channels
    """
    def get_dataset_piece(self, movie, trial, channel):
        return self.all_lfp_data[movie, trial, channel, :]

    """
    Plots signal ...
    """
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


if __name__ == '__main__':
    dataset = CatLFP()
    print(dataset.all_lfp_data.shape)
