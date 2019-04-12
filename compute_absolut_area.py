import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform

from datasets.DATASET_PATHS import GABI_MOUSEACH_DATASET_PATH as MOUSEACH_DATASET_PATH
from datasets.DATASET_PATHS import GABI_MOUSE_DATASET_PATH as MOUSE_DATASET_PATH
from datasets.LFPDataset import LFPDataset
from datasets.MouseControl import MouseControl
from datasets.MouseLFP import MouseLFP


def plot_signal(signal, name):
    plt.figure(figsize=(16, 12))
    plot_title = name
    plt.title(plot_title)
    plt.plot(signal, label="Mouse LFP signal")
    plt.legend()
    plt.savefig(os.path.join("./mean_absolute_area/{0}.png".format(plot_title)))
    # plt.show()


if __name__ == '__main__':
    dataset = LFPDataset(MOUSE_DATASET_PATH)
    signal = [[] for x in range(dataset.number_of_conditions)]
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

    for i in range(0, len(signal)):
        plot_signal(signal[i], "Condition-" + str(i))
    plot_signal(np.sum(signal, axis=0), "AllConditions")
    plot_signal(np.sum(np.sum(trials, axis=1), axis=1) / trials.shape[0], "AllTrials")
    print(np.argmin(np.sum(np.sum(trials, axis=1), axis=1) / trials.shape[0]))