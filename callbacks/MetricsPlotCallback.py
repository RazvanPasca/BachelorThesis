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
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Accuracy')
            # plt.axis([0,100,0,1])
            plt.plot(epochs, self.val_acc, color='r')
            plt.plot(epochs, self.acc, color='b')
            plt.ylabel('accuracy')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if 'loss' in self.metrics:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Loss')
            # plt.axis([0,100,0,5])
            plt.plot(epochs, self.val_loss, color='r')
            plt.plot(epochs, self.loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if "reconstruction_loss" in self.metrics:
            count_subplots += 1
            plt.subplot(self.num_subplots, 1, count_subplots)
            plt.title('Reconstruction_loss')
            # plt.axis([0,100,0,5])
            plt.plot(epochs, self.val_rec_loss, color='r')
            plt.plot(epochs, self.rec_loss, color='b')
            plt.ylabel('loss')

            red_patch = mpatches.Patch(color='red', label='Test')
            blue_patch = mpatches.Patch(color='blue', label='Train')

            plt.legend(handles=[red_patch, blue_patch], loc=4)

        if self.save_graph:
            plt.savefig(self.save_path + '/training_acc_loss.png')
        plt.close()
