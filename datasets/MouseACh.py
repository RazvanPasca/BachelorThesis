from datasets.DATASET_PATHS import MOUSEACH_DATASET_PATH
from datasets.MouseLFP import MouseLFP


class MouseACh(MouseLFP):
    def __init__(self, channels_to_keep=None, val_perc=0.20, test_perc=0.0, random_seed=42, nr_bins=256, nr_of_seqs=3,
                 normalization="Zsc"):
        super().__init__(MOUSEACH_DATASET_PATH, channels_to_keep=None, val_perc=0.20, test_perc=0.0, random_seed=42,
                         nr_bins=256, nr_of_seqs=3,
                         normalization="Zsc")


if __name__ == '__main__':
    dataset = MouseACh()
    print(dataset.all_lfp_data.shape)
