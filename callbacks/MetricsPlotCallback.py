import numpy as np
from keras.callbacks import Callback
from matplotlib import pyplot as plt, patches as mpatches


class MetricsPlotCallback(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph.

    The graph is updated by the 'on_epoch_end' event of the Keras Callback class

    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event

    """

    def __init__(self, save_path, metrics=['acc', 'loss'], save_fig=True):
        self.metrics = metrics
        self.num_subplots = len(metrics)
        self.save_graph = save_fig
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.acc = []
        self.rec_loss = []
        self.val_rec_loss = []
        self.val_acc = []
        self.loss = []
        self.val_loss = []
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_count += 1
        self.val_acc.append(logs.get('val_sparse_categorical_accuracy'))
        self.acc.append(logs.get('sparse_categorical_accuracy'))

        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

        self.rec_loss.append(logs.get("reconstruction_loss"))
        self.val_rec_loss.append(logs.get("val_reconstruction_loss"))
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        plt.figure(figsize=(16, 12))

        if 'acc' in self.metrics:
            count_subplots += 1
            self.plot_test_val_loss("Accuracy", self.acc, self.val_acc, count_subplots, epochs)

        if 'loss' in self.metrics:
            count_subplots += 1
            self.plot_test_val_loss("Loss", self.loss, self.val_loss, count_subplots, epochs)

        if "reconstruction_loss" in self.metrics:
            count_subplots += 1
            self.plot_test_val_loss("Reconstruction_loss", self.rec_loss, self.val_loss, count_subplots, epochs)

        if "kl_loss" in self.metrics:
            count_subplots += 1
            kl_val_loss = np.array(self.val_loss) - np.array(self.val_rec_loss)
            kl_loss = np.array(self.loss) - np.array(self.rec_loss)

            self.plot_test_val_loss("KL_loss", kl_loss, kl_val_loss, count_subplots, epochs)

        if self.save_graph:
            plt.savefig(self.save_path + '/training_acc_loss.png')
        plt.close()

    def plot_test_val_loss(self, title, train_metric, val_metric, count_subplots, epochs):
        plt.subplot(self.num_subplots, 1, count_subplots)
        plt.title(title)
        plt.plot(epochs, val_metric, color='r')
        plt.plot(epochs, train_metric, color='b')
        plt.ylabel(title)
        plt.xlabel("epochs")
        plt.tight_layout()
        red_patch = mpatches.Patch(color='red', label='Test')
        blue_patch = mpatches.Patch(color='blue', label='Train')
        plt.legend(handles=[red_patch, blue_patch], loc=1)
