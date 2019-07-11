import cv2
import keract
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

from model_test_utils.test_ae_model import test_ae_model
from train import TrainingConfiguration
from utils.plot_utils import plot_samples
from utils.system_utils import create_dir_if_not_exists


def test_vae_model(model, params: TrainingConfiguration):
    test_dir = "{}/Test".format(params.model_path)
    create_dir_if_not_exists(test_dir)
    test_ae_model(model, params)

    train_batch = params.dataset.get_train_plot_examples(300)
    plot_latent_space(model, train_batch, test_dir, "PCA plot after train", show=True)


def test_vae_decoder(decoder, params: TrainingConfiguration):
    test_dir = "{}/Test".format(params.model_path)
    create_dir_if_not_exists(test_dir)

    interp_1 = get_vae_interpolation_dist(params.generative_samples, params.z_dim, -2, 2)
    sample_1 = get_vae_sample_dist(params.nr_rec * 2, params.z_dim)

    generate_plot_samples(decoder, sample_1, test_dir, "Generative Samples after train", show=True)
    generate_plot_samples(decoder, interp_1, test_dir, "Interpolation Samples after train", show=True)


def plot_latent_space(model, batch, save_path, name, show=False):
    train_latents = keract.get_activations(model, batch[0], "Z_mean")

    pca = PCA(n_components=2)
    train_latents = next(iter(train_latents.values()))
    comps = pca.fit_transform(train_latents)

    plt.figure(figsize=(15, 15))
    fig, ax = plt.subplots()
    imscatter(comps[:, 0], comps[:, 1], ax, batch[1], 0.4)

    ax.set_title(name)
    plt.savefig("{}/{}.png".format(save_path, name), format="png")
    if show:
        plt.show()
    plt.close()


def generate_plot_samples(model, samples_seed, save_path, name, show=False):
    generated_images = model.predict(samples_seed)
    generated_images = np.squeeze(generated_images, axis=3)
    plot_samples(generated_images, save_path, name, show=show)


def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i] * 255.
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


def get_vae_interpolation_dist(nr_interpolation_samples, z_dim, left_lim=-2, right_lim=2):
    sampler = np.linspace(left_lim, right_lim, nr_interpolation_samples)
    sampler = np.expand_dims(sampler, axis=1)
    first_point = np.random.normal(0, 1, z_dim)
    second_point = np.random.normal(0, 1, z_dim)
    return sampler * first_point + (1 - sampler) * second_point


def get_vae_sample_dist(nr_samples, z_dim):
    return np.random.normal(0, 1, (nr_samples, z_dim)).reshape(
        (nr_samples, z_dim))
