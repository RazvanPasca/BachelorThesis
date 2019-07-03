import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback


class VaeCallback(Callback):
    def __init__(self, model_args):
        self.nr_samples = model_args.generative_samples
        self.epoch = 0
        self.logging_period = model_args.logging_period
        self.generator = model_args.generator
        self.z_dim = model_args.z_dim
        self.save_path = model_args.model_path

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.logging_period == 0:
            self.generate_new_samples()

        self.epoch += 1

    def generate_new_samples(self):

        sampler = np.linspace(-15, 15, self.nr_samples)

        first_point = np.random.normal(0, 1, self.z_dim)
        second_point = np.random.normal(0, 1, self.z_dim)

        interpolations = np.sqrt(sampler) * first_point + np.sqrt(100 - sampler) * second_point

        generated_images = self.generator.predict(interpolations)

        nr_cols = np.sqrt(self.nr_samples)
        nr_rows = nr_cols
        fig, subplots = plt.subplots(nr_rows, nr_cols, sharex=True, figsize=(20, 20))
        for i, subplot in enumerate(subplots.flatten()):
            subplot.imshow(generated_images[i], cmap="gray")
            subplot.axis("off")

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.title("Generated samples")
        plt.savefig("{}/Epoch:{}.png".format(self.save_path, "Generated samples"), format="png")
        plt.show()
        plt.close()
