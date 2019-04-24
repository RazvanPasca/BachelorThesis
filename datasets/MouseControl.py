from datasets.DATASET_PATHS import PASCA_MOUSE_DATASET_PATH
from datasets.MouseLFP import MouseLFP


class MouseControl(MouseLFP):
    def __init__(self, channels_to_keep=(-1,),
                 conditions_to_keep=(-1,),
                 trials_to_keep=(-1,),
                 val_perc=0.20,
                 random_seed=42,
                 nr_bins=256, nr_of_seqs=6, normalization="Zsc", cutoff_freq=50, white_noise_dev=-1):
        super().__init__(RIST_MOUSE_DATASET_PATH,
                         channels_to_keep=channels_to_keep,
                         conditions_to_keep=conditions_to_keep,
                         trials_to_keep=trials_to_keep,
                         val_perc=val_perc,
                         random_seed=random_seed,
                         nr_bins=nr_bins,
                         nr_of_seqs=nr_of_seqs,
                         normalization=normalization,
                         cutoff_freq=cutoff_freq,
                         white_noise_dev=white_noise_dev)


if __name__ == '__main__':
    dataset = MouseControl()
    print(dataset.all_lfp_data.shape)
