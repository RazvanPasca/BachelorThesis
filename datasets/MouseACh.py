
from datasets.DATASET_PATHS import RIST_MOUSEACH_DATASET_PATH

from datasets.MouseLFP import MouseLFP


class MouseACh(MouseLFP):
    def __init__(self, channels_to_keep=None, conditions_to_keep=None, val_perc=0.20, test_perc=0.0, random_seed=42,
                 nr_bins=256, nr_of_seqs=6,
                 normalization="Zsc", cutoff_freq=50):
        super().__init__(RIST_MOUSEACH_DATASET_PATH,
                         channels_to_keep=channels_to_keep,
                         conditions_to_keep=conditions_to_keep,
                         val_perc=val_perc,
                         test_perc=test_perc,
                         random_seed=random_seed,
                         nr_bins=nr_bins,
                         nr_of_seqs=nr_of_seqs,
                         normalization=normalization,
                         cutoff_freq=cutoff_freq)


if __name__ == '__main__':
    dataset = MouseACh()
    print(dataset.all_lfp_data.shape)
