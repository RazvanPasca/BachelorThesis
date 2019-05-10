import os
import numpy as np

from matplotlib import pyplot as plt
from datasets.paths import MOUSEACH_DATASET_PATH
from datasets.paths import MOUSE_DATASET_PATH
from datasets.paths import CAT_DATASET_PATH
from datasets.LFPDataset import LFPDataset


def plot_signal(signal, name, path):
    if not os.path.exists(path):
        os.makedirs(path)
    plt.figure(figsize=(16, 12))
    plot_title = name
    plt.title(plot_title)
    plt.plot(signal, label="Mouse LFP signal")
    plt.legend()
    plt.savefig(os.path.join(path, "{0}.png".format(plot_title)))
    # plt.show()


def load_dataset(path):
    dataset = LFPDataset(path, normalization='Zsc')
    signal = [[] for _ in range(dataset.number_of_conditions)]
    trials = []
    for stimulus_condition in dataset.stimulus_conditions:
        index = int(stimulus_condition['Trial']) - 1
        condition_number = int(stimulus_condition['Condition number']) - 1
        events = [{'timestamp': dataset.event_timestamps[4 * index + i],
                   'code': dataset.event_codes[4 * index + i]} for i in range(4)]
        trial = dataset.channels[:, events[1]['timestamp']:(events[1]['timestamp'] + 2672)]
        signal[condition_number].append(np.sum(np.abs(trial)) / (trial.shape[1] * trial.shape[0]))
        trials.append(trial)
    trials = np.array(trials)

    return trials, signal


def compute_absolute_area_plots(trials, signal, path):
    for i in range(0, len(signal)):
        plot_signal(signal[i], "Condition-" + str(i), path)

    plot_signal(np.sum(signal, axis=0), "AllConditions", path)

    plot_signal(np.sum(np.sum(trials, axis=1), axis=1) / trials.shape[0], "AllTrials", path)


def compute_absolute_area_for_dataset(path):
    output_path = "./absolute_area"
    if path == MOUSE_DATASET_PATH:
        output_path += "/mouse"
    elif path == MOUSEACH_DATASET_PATH:
        output_path += "/mouse_ach"

    trials, signal = load_dataset(path)
    compute_absolute_area_plots(trials, signal, output_path)


if __name__ == '__main__':
    compute_absolute_area_for_dataset(MOUSE_DATASET_PATH)
    compute_absolute_area_for_dataset(MOUSEACH_DATASET_PATH)
