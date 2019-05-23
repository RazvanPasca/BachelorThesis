import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from keras.callbacks import Callback

from output_utils import compute_conf_matrix, log_metrics_to_text, plot_conf_matrix


class AccLossPlotter(Callback):
    """Plot training Accuracy and Loss values on a Matplotlib graph.

    The graph is updated by the 'on_epoch_end' event of the Keras Callback class

    # Arguments
        graphs: list with some or all of ('acc', 'loss')
        save_graph: Save graph as an image on Keras Callback 'on_train_end' event

    """

    def __init__(self, save_path, graphs=['acc', 'loss'], save_fig=True):
        self.graphs = graphs
        self.num_subplots = len(graphs)
        self.save_graph = save_fig
        self.save_path = save_path

    def on_train_begin(self, logs={}):
        self.acc = []
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
        epochs = [x for x in range(self.epoch_count)]

        count_subplots = 0
        plt.figure(figsize=(16, 12))

        if 'acc' in self.graphs:
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

        if 'loss' in self.graphs:
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

        if self.save_graph:
            plt.savefig(self.save_path + '/training_acc_loss.png')
        plt.close()

    def on_train_end(self, logs={}):
        if self.save_graph:
            plt.savefig(self.save_path + '/training_acc_loss.png')


class ConfusionMatrixPlotter(Callback):
    """Plot the confusion matrix on a graph and update after each epoch

    # Arguments
        X_val: The input values
        Y_val: The expected output values
        classes: The categories as a list of string names
        normalize: True - normalize to [0,1], False - keep as is
        cmap: Specify matplotlib colour map
        title: Graph Title

    """

    def __init__(self, X_train, Y_train, X_val, Y_val, classes, save_path, logging_period, normalize=False,
                 cmap=plt.cm.Blues,
                 title='Confusion Matrix', ):
        self.X_val = X_val
        self.Y_val = Y_val
        self.X_train = X_train
        self.Y_train = Y_train
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        self.save_path = save_path
        self.epoch = 0
        self.logging_period = logging_period

        plt.title(self.title)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        """Val conf matrix"""
        if self.epoch % self.logging_period == 0:
            metrics, cnf_mat = compute_conf_matrix(self.model, self.X_val, self.Y_val)
            plot_conf_matrix(cnf_mat, self.classes, self.cmap, self.normalize,
                             self.save_path + "/" + "E:{}_val_conf_matrix.png".format(self.epoch))
            log_metrics_to_text(metrics, self.classes,
                                fname=self.save_path + "/" + "E:{}_val_precision_recall_f1.txt".format(self.epoch))

            """Train conf matrix"""
            metrics, cnf_mat = compute_conf_matrix(self.model, self.X_train, self.Y_train)
            log_metrics_to_text(metrics, self.classes,
                                fname=self.save_path + "/" + "E:{}_train_precision_recall_f1.txt".format(self.epoch))
            plot_conf_matrix(cnf_mat, self.classes, self.cmap, self.normalize,
                             self.save_path + "/" + "E:{}_train_conf_matrix.png".format(self.epoch))
        self.epoch += 1
