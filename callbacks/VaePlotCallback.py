import cv2
import keract
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA

from utils.plot_utils import plot_samples
from utils.system_utils import create_dir_if_not_exists


class VaeCallback(Callback):
    def __init__(self, train_batch, val_batch, model_args):
        self.nr_interpolation_samples = model_args.generative_samples
        self.nr_samples = 2 * model_args.nr_rec
        self.epoch = 0
        self.train_batch = train_batch
        self.val_batch = val_batch
        self.logging_period = model_args.logging_period
        self.decoder = model_args.decoder
        self.z_dim = model_args.z_dim
        self.save_path = "{}/Samples".format(model_args.model_path)
        create_dir_if_not_exists(self.save_path)
        self.get_interpolations_seed()
        self.get_samples_seed()

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.logging_period == 0:
            get_interpolations_samples(self.decoder, self.interpolations_seed_2,
                                       self.save_path,
                                       "Interpolations:2-Epoch:{}".format(self.epoch))
            get_interpolations_samples(self.decoder, self.interpolations_seed_1,
                                       self.save_path,
                                       "Interpolations:1-Epoch:{}".format(self.epoch))

            generate_samples(self.decoder, self.samples_seed_11, self.save_path,
                             "Random_Samples:11-Epoch{}".format(self.epoch))

            generate_samples(self.decoder, self.samples_seed_11, self.save_path,
                             "Random_Samples:1-Epoch{}".format(self.epoch))
        if self.epoch % self.logging_period * 2 == 0:
            plot_latent_space(self.model, self.train_batch, self.save_path,
                              "2D Train latent space visualization epoch:{}".format(self.epoch))
            plot_latent_space(self.model, self.val_batch, self.save_path,
                              "2D Test latent space visualization:{}".format(self.epoch))

        self.epoch += 1

    def get_interpolations_seed(self):
        sampler = np.linspace(-2, 2, self.nr_interpolation_samples)
        sampler = np.expand_dims(sampler, axis=1)

        first_point = np.random.normal(0, 1, self.z_dim)
        second_point = np.random.normal(0, 1, self.z_dim)

        self.interpolations_seed_2 = sampler * first_point + (1 - sampler) * second_point

        sampler = np.linspace(-1, 1, self.nr_interpolation_samples)
        sampler = np.expand_dims(sampler, axis=1)

        first_point = np.random.normal(0, 1, self.z_dim)
        second_point = np.random.normal(0, 1, self.z_dim)

        self.interpolations_seed_1 = sampler * first_point + (1 - sampler) * second_point

    def get_samples_seed(self):
        self.samples_seed_1 = np.random.normal(0, 1, (self.nr_samples, self.z_dim)).reshape(
            (self.nr_samples, self.z_dim))

        self.samples_seed_11 = np.random.normal(0, 1, (self.nr_samples, self.z_dim)).reshape(
            (self.nr_samples, self.z_dim))


def plot_latent_space(model, batch, save_path, name):
    train_latents = keract.get_activations(model, batch[0], "Z_mean")
    labels = ["Condition 0", "Condition 1", "Condition 2"]

    pca = PCA(n_components=2)
    train_latents = next(iter(train_latents.values()))
    comps = pca.fit_transform(train_latents)

    #
    # for cond in range(3):
    #     cond_indices = []
    #     for i, seq_addr in enumerate(batch[2]):
    #         if seq_addr.condition == cond:
    #             cond_indices.append(i)
    # plt.scatter(comps[cond_indices, 0], comps[cond_indices, 1], label=labels[cond])

    plt.figure(figsize=(15, 15))
    fig, ax = plt.subplots()
    imscatter(comps[:, 0], comps[:, 1], ax, batch[1], 0.4)

    plt.title(name)
    plt.savefig("{}/{}.png".format(save_path, name), format="png")
    plt.show()
    plt.close()


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


def get_interpolations_samples(model, interp_seed, save_path, name):
    generated_images = model.predict(interp_seed)
    generated_images = np.squeeze(generated_images, axis=3)

    plot_samples(generated_images, save_path, name)


def generate_samples(model, samples_seed, save_path, name):
    generated_images = model.predict(samples_seed)
    generated_images = np.squeeze(generated_images, axis=3)

    plot_samples(generated_images, save_path, name)
