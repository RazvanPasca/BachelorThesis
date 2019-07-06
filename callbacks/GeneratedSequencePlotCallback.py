import tensorflow as tf
from keras import callbacks

from train import TrainingConfiguration
from utils.plot_utils import generate_multi_plot


class GeneratedSequencePlotCallback(callbacks.Callback):
    """Callback used for plotting at certain epochs the generations of the model

        If nr_predictions is smaller than 1, the model will predict up to the end of the signal

        If starting point is smaller than 1, the model will start predicting from the beginning of the signal-frame size
    """

    def __init__(self, model_parameters: TrainingConfiguration, plot_period, nr_predictions, starting_point,
                 all_reset_indices,
                 nr_plot_rows=3):
        super().__init__()

        self.model_params = model_parameters
        self.epoch = 0
        self.get_nr_prediction_steps(model_parameters, nr_predictions, starting_point)
        self.plot_period = plot_period
        self.starting_point = starting_point
        self.all_reset_indices = self.get_all_reset_indices(all_reset_indices)
        self.nr_plot_rows = nr_plot_rows
        self.nr_of_sequences_to_plot = self.model_params.nr_rec // nr_plot_rows * nr_plot_rows if \
            self.model_params.nr_rec > nr_plot_rows else self.model_params.nr_rec
        self.all_pred_losses_normalized = {"VAL": [],
                                           "TRAIN": []}

    def get_all_reset_indices(self, all_reset_indices):
        set_all_reset_indices = [set(tuple(reset_indices)) for reset_indices in all_reset_indices]
        return set_all_reset_indices

    def set_model(self, model):
        self.pred_error_writer = tf.summary.FileWriter(self.model_params.model_path)
        super().set_model(model)

    def on_epoch_end(self, epoch, logs={}):
        self.epoch += 1

        if self.epoch % self.plot_period == 0 or self.epoch == 1:
            all_pred_losses_normalized_mean = generate_multi_plot(self.model, self.model_params, self.epoch,
                                                                  self.starting_point,
                                                                  nr_prediction_steps=self.nr_prediction_steps,
                                                                  all_reset_indices=self.all_reset_indices,
                                                                  nr_sequences_to_plot=self.nr_of_sequences_to_plot,
                                                                  nr_rows=self.nr_plot_rows,
                                                                  source="TRAIN")
            self.write_pred_losses_to_tboard(all_pred_losses_normalized_mean, self.epoch)
            self.update_all_pred_losses_mean(all_pred_losses_normalized_mean)

            all_pred_losses_normalized_mean = generate_multi_plot(self.model, self.model_params, self.epoch,
                                                                  self.starting_point,
                                                                  nr_prediction_steps=self.nr_prediction_steps,
                                                                  all_reset_indices=self.all_reset_indices,
                                                                  nr_sequences_to_plot=self.nr_of_sequences_to_plot,
                                                                  nr_rows=self.nr_plot_rows,
                                                                  source="VAL")
            self.write_pred_losses_to_tboard(all_pred_losses_normalized_mean, self.epoch)

            self.update_all_pred_losses_mean(all_pred_losses_normalized_mean)

    def on_train_end(self, logs=None):
        all_pred_losses_normalized = generate_multi_plot(self.model, self.model_params, "TrainEnd", self.starting_point,
                                                         nr_prediction_steps=self.nr_prediction_steps,
                                                         all_reset_indices=self.all_reset_indices,
                                                         nr_sequences_to_plot=self.nr_of_sequences_to_plot,
                                                         nr_rows=self.nr_plot_rows)

        self.update_all_pred_losses_mean(all_pred_losses_normalized)
        self.write_pred_losses_to_tboard(all_pred_losses_normalized, self.epoch)
        self.pred_error_writer.close()

    def get_nr_prediction_steps(self, model_params, nr_predictions, starting_point):
        self.nr_prediction_steps = nr_predictions if nr_predictions > 0 else model_params.dataset.trial_length
        self.nr_prediction_steps = min(self.nr_prediction_steps,
                                       model_params.dataset.trial_length - starting_point - model_params.slice_length - 1)

    def write_pred_losses_to_tboard(self, all_pred_losses_normalized, epoch):
        for source, source_errors in all_pred_losses_normalized.items():
            summary = tf.Summary()
            for i, error in enumerate(source_errors):
                summary_value = summary.value.add()
                summary_value.simple_value = error
                summary_value.tag = "{}_Norm_Pred_Error_GenWSize:{}".format(source,
                                                                            list(self.all_reset_indices[i])[:10])
            self.pred_error_writer.add_summary(summary, epoch)
        self.pred_error_writer.flush()

    def update_all_pred_losses_mean(self, all_pred_losses_normalized):
        for source, source_pred_errors in all_pred_losses_normalized.items():
            self.all_pred_losses_normalized[source].append(source_pred_errors)
