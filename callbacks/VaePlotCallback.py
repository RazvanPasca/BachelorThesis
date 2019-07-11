from keras.callbacks import Callback

from model_test_utils.test_vae_model import get_vae_interpolation_dist, get_vae_sample_dist, plot_latent_space, \
    generate_plot_samples
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
        self.decoder_path = model_args.model_path
        create_dir_if_not_exists(self.save_path)
        self.get_interpolations_seed()
        self.get_samples_seed()

    def on_epoch_end(self, epoch, logs={}):
        if self.epoch % self.logging_period == 0:
            generate_plot_samples(self.decoder, self.interpolations_seed_2,
                                  self.save_path,
                                  "Interpolations:2-Epoch:{}".format(self.epoch))
            generate_plot_samples(self.decoder, self.interpolations_seed_1,
                                  self.save_path,
                                  "Interpolations:1-Epoch:{}".format(self.epoch))

            generate_plot_samples(self.decoder, self.samples_seed_11, self.save_path,
                                  "Random_Samples:11-Epoch{}".format(self.epoch))

            generate_plot_samples(self.decoder, self.samples_seed_11, self.save_path,
                                  "Random_Samples:1-Epoch{}".format(self.epoch))
            self.decoder.save("{}/decoder.h5".format(self.decoder_path))

        if self.epoch % self.logging_period * 2 == 0:
            plot_latent_space(self.model, self.train_batch, self.save_path,
                              "2D Train latent space visualization epoch:{}".format(self.epoch))
            plot_latent_space(self.model, self.val_batch, self.save_path,
                              "2D Test latent space visualization epoch:{}".format(self.epoch))

        self.epoch += 1

    def get_interpolations_seed(self):
        self.interpolations_seed_2 = get_vae_interpolation_dist(self.nr_interpolation_samples,
                                                                self.z_dim, -2, 2)

        self.interpolations_seed_1 = get_vae_interpolation_dist(self.nr_interpolation_samples,
                                                                self.z_dim, - 1, 1)

    def get_samples_seed(self):
        self.samples_seed_1 = get_vae_sample_dist(self.nr_samples, self.z_dim)

        self.samples_seed_11 = get_vae_sample_dist(self.nr_samples, self.z_dim)
