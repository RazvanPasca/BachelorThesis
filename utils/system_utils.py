import csv
import os

import numpy as np


def create_dir_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def prepare_file_for_writing(file_path, text):
    with open(file_path, "w") as f:
        f.write(text)


def log_range_accumulated_errors(epoch, prediction_losses, reset_indices, save_path):
    csv_name = os.path.join(save_path, "prediction_losses_means.csv")

    if os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            errors_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            errors_writer.writerow(np.insert(prediction_losses, 0, epoch))

    else:
        with open(csv_name, 'w') as f:
            errors_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            errors_writer.writerow(["Epoch"] + list(reset_indices))
            errors_writer.writerow(np.insert(prediction_losses, 0, epoch))


def log_metrics_to_text(metrics, classes, fname):
    formatter = ",".join("{:25}" for _ in range(len(metrics) - 1)) + "\n"
    with open(fname, "w+") as f:
        keys = list(metrics.keys())
        f.write(formatter.format("Class", *keys[:-2]), )
        for i in range(len(classes)):
            f.write(formatter.format(classes[i], *[metrics[key][i] for key in keys[:-2]]))
        f.write(str([(key, val) for key, val in list(metrics.items())[-2:]]))
