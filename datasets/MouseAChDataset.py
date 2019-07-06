from datasets.LFPDataset import LFPDataset
from datasets.datasets_utils import SplitStrategy, SlicingStrategy, ModelType
from datasets.paths import MOUSE_ACH_DATASET_SIGNAL_PATH, MOUSE_ACH_DATASET_STIMULI_PATH


class MouseAChDataset(LFPDataset):
    def __init__(self,
                 val_percentage=0.00,
                 split_by=SplitStrategy.TRIALS,
                 random_seed=42,
                 orientations_to_keep=None,
                 contrasts_to_keep=None,
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

        if orientations_to_keep is None:
            orientations_to_keep = list(range(8))

        if contrasts_to_keep is None:
            contrasts_to_keep = list(range(3))

        conditions_to_keep = [o * 3 + c for o in orientations_to_keep for c in contrasts_to_keep]

        super().__init__(
            signal_path=MOUSE_ACH_DATASET_SIGNAL_PATH,
            stimuli_path=MOUSE_ACH_DATASET_STIMULI_PATH,
            val_percentage=val_percentage,
            split_by=split_by,
            random_seed=random_seed,
            conditions_to_keep=conditions_to_keep,
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

        self.orientations_to_keep = orientations_to_keep
        self.contrasts_to_keep = contrasts_to_keep
