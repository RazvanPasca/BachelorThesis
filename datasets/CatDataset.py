import matplotlib.pyplot as plt

from datasets.LFPDataset import LFPDataset
from datasets.datasets_utils import SplitStrategy, SlicingStrategy, ModelType
from datasets.paths import CAT_DATASET_SIGNAL_PATH, CAT_DATASET_STIMULI_PATH_64


class CatDataset(LFPDataset):
    def __init__(self,
                 val_percentage=0.0,
                 split_by=SplitStrategy.TRIALS,
                 random_seed=42,
                 conditions_to_keep=None,
                 trials_to_keep=None,
                 channels_to_keep=None,
                 slice_length=1000,
                 slicing_strategy=SlicingStrategy.RANDOM,
                 model_type=ModelType.NEXT_TIMESTEP,
                 cutoff_freq=None,
                 stack_channels=False,
                 use_mu_law=False,
                 number_of_bins=255,
                 condition_on_gamma=False,
                 gamma_windows_in_trial=None,
                 blur_images=False):
        self.conditions_to_keep = conditions_to_keep
        super().__init__(
            signal_path=CAT_DATASET_SIGNAL_PATH,
            stimuli_path=CAT_DATASET_STIMULI_PATH_64,
            val_percentage=val_percentage,
            split_by=split_by,
            random_seed=random_seed,
            conditions_to_keep=self.conditions_to_keep,
            trials_to_keep=trials_to_keep,
            channels_to_keep=channels_to_keep,
            slice_length=slice_length,
            slicing_strategy=slicing_strategy,
            model_type=model_type,
            cutoff_freq=cutoff_freq,
            stack_channels=stack_channels,
            use_mu_law=use_mu_law,
            number_of_bins=number_of_bins,
            condition_on_gamma=condition_on_gamma,
            gamma_windows_in_trial=gamma_windows_in_trial,
            blur_images=blur_images)

    def plot_stimuli_hist(self, path):
        """
        Plots stimuli histogram.
        """

        plt.figure(figsize=(16, 12))
        plt.hist([movie for movie in self.regression_actual], bins=30,
                 label=["Movie:{}".format(i) for i in self.conditions_to_keep])
        plt.legend(prop={'size': 10})
        plt.title("Histogram of {} with mean {:10.4f} and std {:10.4f}".format(self.model_type,
                                                                               self.regression_target_mean,
                                                                               self.regression_target_std))
        plt.savefig("{}/movies_regression_histo.png".format(path), format="png")
