import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from datasets.DATASET_PATHS import PASCA_CAT_DATASET_PATH as CAT_DATASET_PATH
from datasets.DATASET_PATHS import PASCA_MOUSE_DATASET_PATH as MOUSE_DATASET_PATH
from datasets.DATASET_PATHS import PASCA_MOUSEACH_DATASET_PATH as MOUSEACH_DATASET_PATH
from datasets.LFPDataset import LFPDataset


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def compute_similarity_matrix(ds):
    row = []
    for i, channel1 in enumerate(ds.channels):
        column = []
        for j, channel2 in enumerate(ds.channels):
            column.append(rmse(channel1, channel2))
        row.append(np.array(column))
    return np.array(row)


def compute_heatmap_on_dataset(ds):
    sim_matrix = compute_similarity_matrix(ds)
    sns.set()
    sns.heatmap(sim_matrix)
    plt.show()
    return sim_matrix


def compute_signals_correlation(ds, name):
    sim_matrix = compute_similarity_matrix(ds)
    means = np.mean(sim_matrix, axis=0)
    corr_indexes_sorted = np.argsort(means)

    with open("{}:Signal_correlations.txt".format(name), "w+") as f:
        for index in corr_indexes_sorted[::-1]:
            f.write("Channel:{}, Mean rmse: {}\n".format(index, means[index]))


if __name__ == '__main__':
    dataset = LFPDataset(CAT_DATASET_PATH)
    compute_signals_correlation(dataset, "CAT")

    dataset = LFPDataset(MOUSE_DATASET_PATH)
    compute_signals_correlation(dataset, "MOUSE Control")

    dataset = LFPDataset(MOUSEACH_DATASET_PATH)
    compute_signals_correlation(dataset, "MOUSE ACh")
