import itertools

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


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
        self.val_acc.append(logs.get('val_acc'))
        self.acc.append(logs.get('acc'))
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

    def __init__(self, X_val, Y_val, classes, save_path, normalize=False, cmap=plt.cm.Blues, title='Confusion Matrix'):
        self.X_val = X_val
        self.Y_val = Y_val
        self.title = title
        self.classes = classes
        self.normalize = normalize
        self.cmap = cmap
        self.save_path = save_path

        plt.title(self.title)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.X_val)
        max_pred = np.argmax(pred, axis=1)
        max_y = self.Y_val
        cnf_mat = confusion_matrix(max_y, max_pred)
        plt.figure(figsize=(16, 12))

        if self.normalize:
            cnf_mat = cnf_mat.astype('float') / cnf_mat.sum(axis=1)[:, np.newaxis]

        thresh = cnf_mat.max() / 2.
        for i, j in itertools.product(range(cnf_mat.shape[0]), range(cnf_mat.shape[1])):
            plt.text(j, i, cnf_mat[i, j],
                     horizontalalignment="center",
                     color="white" if cnf_mat[i, j] > thresh else "black")

        plt.imshow(cnf_mat, interpolation='nearest', cmap=self.cmap)

        # Labels
        # tick_marks = np.arange(len(self.classes))
        # plt.xticks(tick_marks, self.classes, rotation=45)
        # plt.yticks(tick_marks, self.classes)

        predicted_positives = np.sum(cnf_mat, axis=0)
        actual_positives = np.sum(cnf_mat, axis=1)

        precision = np.diag(cnf_mat) / predicted_positives
        recall = np.diag(cnf_mat) / actual_positives

        f1 = 2 * (precision * recall) / (precision + recall + 0.00001)

        formatter = "{:10}|{:10.4}|{:10.4}|{:10.4}\n"
        with open(self.save_path + "/" + "precision_recall_f1.txt", "w+") as f:
            f.write(formatter.format("Class", "Precision", "Recall", "F1"))
            for i in range(precision.shape[0]):
                f.write(formatter.format(i, precision[i], recall[i], f1[i]))

        plt.colorbar()

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        # plt.draw()
        plt.savefig(self.save_path + "/" + "conf_matrix.png")
        plt.close()
