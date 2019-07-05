import numpy as np
from keras.callbacks import Callback

from utils.plot_utils import plot_samples
from utils.system_utils import create_dir_if_not_exists


class VaeCallback(Callback):
    def __init__(self, model_args):
        self.nr_interpolation_samples = model_args.generative_samples
        self.nr_samples = 2 * model_args.nr_rec
        self.epoch = 0
        self.logging_period = model_args.logging_period
        self.generator = model_args.generator
        print(self.generator.summary())
        self.z_dim = model_args.z_dim
        self.save_path = "{}/Samples".format(model_args.model_path)
        create_dir_if_not_exists(self.save_path)
        self.get_interpolations_seed()
        self.get_samples_seed()

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.logging_period == 0:
            self.samples_interpolation()
            self.generate_samples()

        self.epoch += 1

    def get_interpolations_seed(self):
        sampler = np.linspace(-5, 5, self.nr_interpolation_samples)
        sampler = np.expand_dims(sampler, axis=1)

        first_point = np.random.normal(0, 1, self.z_dim)
        second_point = np.random.normal(0, 1, self.z_dim)

        self.interpolations_seed = sampler * first_point + (self.nr_interpolation_samples - sampler) * second_point

    def get_samples_seed(self):
        self.samples_seed = np.random.normal(0, 2, (self.nr_samples, self.z_dim)).reshape((self.nr_samples, self.z_dim))

    def samples_interpolation(self):
        generated_images = self.generator.predict(self.interpolations_seed)
        generated_images = np.squeeze(generated_images, axis=3)

        plot_samples(generated_images, self.save_path, "Interpolations", self.epoch)

    def generate_samples(self):
        generated_images = self.generator.predict(self.samples_seed)
        generated_images = np.squeeze(generated_images, axis=3)

        plot_samples(generated_images, self.save_path, "Random Samples", self.epoch)
