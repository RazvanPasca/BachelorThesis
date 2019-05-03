import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from datasets.DATASET_PATHS import PASCA_MOUSEACH_DATASET_PATH, PASCA_MOUSE_DATASET_PATH
from datasets.MouseLFP import MouseLFP


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
    print(x, values_over_x)
    return values_over_x


def compare_signals(dataset, dir_name):
    conditions = len(dataset.conditions_to_keep)
    trials = len(dataset.trials_to_keep)
    channels = len(dataset.channels_to_keep)

    for condition in range(conditions):
        for trial in range(trials):
            series = []
            for channel in range(5, 10):
                signal = dataset.get_dataset_piece(condition, trial, channel)
                series.append(signal)

            plt.figure(figsize=(16, 12))
            for i, signal in enumerate(series):
                title = "Condition:{}_Trial:{}".format(condition, trial)
                plt.title(title)
                plt.ylim(-8, +8)
                plt.plot(signal, label=i)
            plt.legend()
            # plt.savefig(
            #     fname="/home/pasca/School/Licenta/Naturix/Correlations/{}/Channel_correlation/{}.png".format(dir_name,
            #                                                                                                  title),
            #     format="png")
            plt.show()
            plt.close()
    print("Finished channel comparison")

    for condition in range(conditions):
        for channel in range(channels):
            series = []
            for trial in range(trials):
                signal = dataset.get_dataset_piece(condition, trial, channel)
                series.append(signal)

            plt.figure(figsize=(16, 12))
            for i, signal in enumerate(series):
                title = "Condition:{}_Channel:{}".format(condition, channel)
                plt.title(title)
                plt.ylim(-5, +5)
                plt.plot(signal, label=i)
            plt.legend()
            plt.savefig(
                fname="/home/pasca/School/Licenta/Naturix/Correlations/{}/Trial_correlation/{}.png".format(dir_name,
                                                                                                           title),
                format="png")
            # plt.show()
            plt.close()

    print("Finished trial comparison")
    for trial in range(trials):
        for channel in range(channels):
            series = []
            for condition in range(conditions):
                signal = dataset.get_dataset_piece(condition, trial, channel)
                series.append(signal)

            plt.figure(figsize=(16, 12))
            for i, signal in enumerate(series):
                title = "Trial:{}_Channel:{}".format(trial, channel)
                plt.title(title)
                plt.ylim(-5, +5)
                plt.plot(signal, label=i)
            plt.legend()
            plt.savefig(
                fname="/home/pasca/School/Licenta/Naturix/Correlations/{}/Condition_correlation/{}.png".format(dir_name,
                                                                                                               title),
                format="png")
            # plt.show()
            plt.close()
    print("Finished condition comparison")


def signal_histograms():
    for normalization in ["MuLaw", "Zsc", "Brute"]:
        dataset = MouseLFP(PASCA_MOUSEACH_DATASET_PATH, cutoff_freq=-1, channels_to_keep=[-1],
                           conditions_to_keep=[0], trials_to_keep=[1], normalization=normalization)
        for cond in range(dataset.number_of_conditions):
            for trial in range(dataset.trials_per_condition):
                signals = []
                for channel in range(33):
                    if channel in dataset.channels_to_keep:
                        signal = dataset.get_dataset_piece(cond, trial, channel)
                        signals.append(signal)

                signals = np.concatenate(signals).ravel()
                a = plt.hist(signals, 512)

                plt.savefig(
                    "/home/pasca/School/Licenta/Naturix/Histograms/MouseACh/{}/Cond:{}_Trial:{}".format(
                        normalization,
                        cond, trial))
                plt.show()
                plt.close()

    for normalization in ["MuLaw", "Zsc", "Brute"]:
        dataset = MouseLFP(PASCA_MOUSE_DATASET_PATH, cutoff_freq=7, channels_to_keep=[-1],
                           conditions_to_keep=[-1], trials_to_keep=[-1], normalization=normalization)
        for cond in range(dataset.number_of_conditions):
            for trial in range(dataset.trials_per_condition):
                signals = []
                for channel in range(32):
                    signal = dataset.get_dataset_piece(cond, trial, channel)
                    signals.append(signal)

                signals = np.concatenate(signals).ravel()
                a = plt.hist(signals, 512)

                plt.savefig(
                    "/home/pasca/School/Licenta/Naturix/Histograms/MouseControl/{}/Cond:{}_Trial:{}".format(
                        normalization,
                        cond, trial))
                plt.close()


if __name__ == '__main__':
    dataset = MouseLFP(PASCA_MOUSE_DATASET_PATH, cutoff_freq=[30, 70], channels_to_keep=[-1],
                       conditions_to_keep=[1], trials_to_keep=[0], normalization="Zsc")
    compare_signals(dataset, "MouseControl/Bpass")
