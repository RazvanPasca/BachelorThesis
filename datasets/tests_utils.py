from datasets.CatDataset import CatDataset
from datasets.datasets_utils import ModelType, SlicingStrategy, SplitStrategy

cat_dataset_arguments = {
    "val_percentage": 0.15,
    "slice_length": 1000,
    "model_type": ModelType.NEXT_TIMESTEP,
    "movies_to_keep": [0, 1, 2],
    "slicing_strategy": SlicingStrategy.RANDOM,
    "stack_channels": True,
    "split_by": SplitStrategy.TRIALS,
    "random_seed": 42
}

BATCH_SIZE = 10


def test_dataset_outputs_with_configuration(dataset_class, expected_output_shapes, configuration):
    dataset = dataset_class(**configuration)
    for x, y in dataset.train_sample_generator(BATCH_SIZE):
        assert (len(x.shape) == len(expected_output_shapes[1]) and expected_output_shapes[1] == x.shape)
        assert (len(y.shape) == len(expected_output_shapes[0]) and expected_output_shapes[0] == y.shape)
        break


def test_cat(args, expected_shape):
    test_dataset_outputs_with_configuration(CatDataset, expected_shape, args)


def test_cat_brightness(args):
    args["model_type"] = ModelType.BRIGHTNESS
    test_cat(args, [(BATCH_SIZE, 1), (BATCH_SIZE, 47, args["slice_length"])])


def test_cat_next_timestep(args):
    args["model_type"] = ModelType.NEXT_TIMESTEP
    features_shape = (BATCH_SIZE, args["slice_length"]) if not args["stack_channels"] else (BATCH_SIZE, 47, args["slice_length"])
    label_shape = (BATCH_SIZE,) if not args["stack_channels"] else (BATCH_SIZE, 47)
    test_cat(args, [label_shape, features_shape])


def test_cat_edges_amount(args):
    args["model_type"] = ModelType.EDGES
    test_cat(args, [(BATCH_SIZE, 1), (BATCH_SIZE, 47, args["slice_length"])])


def test_cat_reconstruction(args):
    args["model_type"] = ModelType.IMAGE_REC
    test_cat(args, [(BATCH_SIZE, 64, 64, 1), (BATCH_SIZE, 47, args["slice_length"])])


def test_cat_condition_classification(args):
    args["model_type"] = ModelType.CONDITION_CLASSIFICATION
    test_cat(args, [(BATCH_SIZE, 1), (BATCH_SIZE, 47, args["slice_length"])])
