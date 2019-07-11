import numpy as np
from matplotlib import pyplot as plt

from train import TrainingConfiguration
from utils.system_utils import create_dir_if_not_exists


def plot_images_reconstructions(images_reconstr, original_images, save_path, name, show=False):
    if images_reconstr.shape[3] == 1:
        images_reconstr = np.squeeze(images_reconstr, axis=3)
    if original_images.shape[3] == 1:
        original_images = np.squeeze(original_images, axis=3)

    nr_cols = 6
    nr_rows = images_reconstr.shape[0] // 6 * 2
    fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(20, 20), num=name)
    for i, subplot_row in enumerate(subplots):
        for j, subplot in enumerate(subplot_row):
            subplot.imshow(
                original_images[i // 2 * nr_cols + j] if i % 2 == 0 else images_reconstr[i // 2 * nr_cols + j],
                cmap="gray")
            subplot.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(name, fontsize=15)
    plt.savefig("{}/Epoch:{}.png".format(save_path, name), format="png")
    if show:
        plt.show()
    plt.close()


def test_ae_model(model, params: TrainingConfiguration):
    train_batch = params.dataset.get_train_plot_examples(params.nr_rec)
    test_batch = params.dataset.get_val_plot_examples(params.nr_rec)
    train_reconstr = model.predict(train_batch[0])
    test_reconstr = model.predict(test_batch[0])

    test_dir = "{}/Test".format(params.model_path)
    create_dir_if_not_exists(test_dir)
    plot_images_reconstructions(train_reconstr, train_batch[1],
                                test_dir, "FULL_TRAIN", show=True)
    plot_images_reconstructions(test_reconstr, test_batch[1],
                                test_dir, "FULL_TEST", show=True)
