import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from datasets.CatLFP import CatLFP
from datasets.DATASET_PATHS import PASCA_MOUSE_DATASET_PATH, PASCA_MOUSE_DATASET_PATH_OLD, PASCA_CAT_DATASET_PATH
from datasets.MouseLFP import MouseLFP
from signal_utils import butter_lowpass_filter


def rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def corr(a, b):
    return np.dot(a, b)


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
    sns.heatmap(sim_matrix, cmap="coolwarm")
    plt.show()
    return sim_matrix


def compute_signals_correlation(ds, name):
    sim_matrix = compute_similarity_matrix(ds)
    means = np.mean(sim_matrix, axis=0)
    corr_indexes_sorted = np.argsort(means)

    with open("{}:Signals_correlations.txt".format(name), "w+") as f:
        for index in corr_indexes_sorted[::-1]:
            f.write("Channel:{}, Cross correlation: {}\n".format(index, means[index]))


def find_samples_over_x(dataset, x):
    samples = dataset.channels[2]
    values_over_x = (samples > x).sum() if x > 0 else (samples < x).sum()
    # print(np.max(samples), np.min(samples))
    print(x, values_over_x)
    return values_over_x


if __name__ == '__main__':
    # dataset = CatLFP(movies_to_keep=[1], channels_to_keep=[0], normalization="Zsc")
    # compute_signals_correlation(dataset, "MOUSE ACh")
    # compute_heatmap_on_dataset(dataset)
    # find_samples_over_x(dataset, 5)
    # find_samples_over_x(dataset, 4)
    # x1 = find_samples_over_x(dataset, 3)
    # find_samples_over_x(dataset, 2)
    # find_samples_over_x(dataset, 1)
    # find_samples_over_x(dataset, 0)
    # find_samples_over_x(dataset, -1)
    # find_samples_over_x(dataset, -2)
    # x2 = find_samples_over_x(dataset, -3)
    # print((x1 + x2) / dataset.channels[2].size)
    dataset = MouseLFP(PASCA_MOUSE_DATASET_PATH, channels_to_keep=[32], cutoff_freq=5)
    series = []

    dataset.plot_signal(18, 1, 32, )

    # for experiment in [13]:
    #     for trial in [9]:
    #         for channel in range(32):
    #             signal = dataset.get_dataset_piece(experiment, trial, channel)
    #             series.append(signal)
    # for i, signal in enumerate(series):
    #     # plt.subplot(2, 1, 1)
    #     plt.ylim(-4, +4)
    #     plt.plot(signal, label=i)
    #     plt.legend()
    # # plt.show()

    # series = []
    # for experiment in [8]:
    #     for trial in [5, 6, 7, 8, 9,]:
    #         signal = dataset.get_dataset_piece(experiment, trial, 4)
    #         series.append(signal)
    # for i, signal in enumerate(series):
    #     plt.subplot(2, 1, 2)
    #     plt.ylim(-4, +4)
    #     plt.plot(signal, label=5 + i)
    #     plt.legend()
    plt.show()

    # for i in range(1):
    #     for j in range(10):
    #         signal = dataset.get_dataset_piece(i, j, 4)
    #         series.append(signal)
    #
    # for signal in series:
    #     plt.plot(signal)

    # find_samples_over_x(dataset, -4)
    # find_samples_over_x(dataset, -5)
    # plt.hist(dataset.channels[2].flatten(), bins=50, range=(-7, 7))
    # plt.show()
