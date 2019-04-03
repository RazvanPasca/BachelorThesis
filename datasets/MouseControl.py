from datasets.DATASET_PATHS import PASCA_MOUSE_DATASET_PATH
from datasets.MouseLFP import MouseLFP


class MouseControl(MouseLFP):
    def __init__(self, channels_to_keep=None, conditions_to_keep=None, val_perc=0.20, test_perc=0.0, random_seed=42,
                 nr_bins=256, nr_of_seqs=3, normalization="Zsc", low_pass_filter=False):
        super().__init__(PASCA_MOUSE_DATASET_PATH,
                         channels_to_keep=channels_to_keep,
                         conditions_to_keep=conditions_to_keep,
                         val_perc=val_perc,
                         test_perc=test_perc,
                         random_seed=random_seed,
                         nr_bins=nr_bins,
                         nr_of_seqs=4,
                         normalization=normalization,
                         low_pass_filter=low_pass_filter)


if __name__ == '__main__':
    dataset = MouseControl()
    print(dataset.all_lfp_data.shape)
