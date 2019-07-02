from datasets.datasets_utils import ModelType, SlicingStrategy, SplitStrategy
from datasets.tests_utils import cat_dataset_arguments, test_cat_next_timestep, test_cat_brightness, \
    test_cat_edges_amount, test_cat_reconstruction, test_cat_condition_classification


def test_cat_next_timestep_one_channel_setup():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.RANDOM
    dataset_args["split_by"] = SplitStrategy.TRIALS
    dataset_args["stack_channels"] = False

    test_cat_next_timestep(dataset_args)
    print("CAT NEXT_TIMESTEP ONE CHANNEL PASSED")


def test_cat_brightness_random_trials():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.RANDOM
    dataset_args["split_by"] = SplitStrategy.TRIALS

    test_cat_brightness(dataset_args)
    print("CAT BRIGHTNESS RANDOM SEQUENCE PASSED")


def test_cat_brightness_consecutive_slices():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.CONSECUTIVE
    dataset_args["split_by"] = SplitStrategy.SLICES

    test_cat_brightness(dataset_args)
    print("CAT BRIGHTNESS SLICED SEQUENCE SPLIT SLICES PASSED")


def test_cat_brightness_consecutive_trials():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.CONSECUTIVE
    dataset_args["split_by"] = SplitStrategy.TRIALS

    test_cat_brightness(dataset_args)
    print("CAT BRIGHTNESS SLICED SEQUENCE SPLIT TRIAL PASSED")


def test_cat_edges_amount_random_trials():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.RANDOM
    dataset_args["split_by"] = SplitStrategy.TRIALS

    test_cat_edges_amount(dataset_args)
    print("CAT EDGES AMOUNT RANDOM TRIALS PASSED PASSED")


def test_cat_edges_amount_consecutive_trials():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.CONSECUTIVE
    dataset_args["split_by"] = SplitStrategy.TRIALS

    test_cat_edges_amount(dataset_args)
    print("CAT EDGES AMOUNT CONSECUTIVE TRIALS PASSED PASSED")


def test_edges_amount_consecutive_slices():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.CONSECUTIVE
    dataset_args["split_by"] = SplitStrategy.SLICES

    test_cat_edges_amount(dataset_args)
    print("CAT EDGES AMOUNT CONSECUTIVE TRIALS SLICES PASSED")


def test_reconstruction_consecutive_trials():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.CONSECUTIVE
    dataset_args["split_by"] = SplitStrategy.SLICES

    test_cat_reconstruction(dataset_args)
    print("CAT RECONSTRUCTION CONSECUTIVE SLICES PASSED")


def test_condition_classification_setup():
    dataset_args = cat_dataset_arguments.copy()
    dataset_args["slicing_strategy"] = SlicingStrategy.RANDOM
    dataset_args["split_by"] = SplitStrategy.TRIALS

    test_cat_condition_classification(dataset_args)
    print("CAT CONDITION CLASSIFICATION RANDOM TRAILS PASSED")


if __name__ == '__main__':
    test_cat_next_timestep_one_channel_setup()
    test_cat_brightness_random_trials()
    test_cat_brightness_consecutive_slices()
    test_cat_brightness_consecutive_trials()
    test_cat_edges_amount_random_trials()
    test_cat_edges_amount_consecutive_trials()
    test_edges_amount_consecutive_slices()
    test_reconstruction_consecutive_trials()
    test_condition_classification_setup()
