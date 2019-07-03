from keras.callbacks import Callback
from matplotlib import pyplot as plt

from utils.output_utils import compute_conf_matrix
from utils.plot_utils import plot_conf_matrix
from utils.system_utils import log_metrics_to_text


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

    def __init__(self, train_batch, val_batch, classes, save_path, logging_period, normalize=False,
                 cmap=plt.cm.Blues,
                 title='Confusion Matrix', ):
        self.X_train, self.Y_train = train_batch
        self.X_val, self.Y_val = val_batch
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
