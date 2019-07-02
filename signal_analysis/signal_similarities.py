import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from datasets.oldies.MouseLFP import MouseLFP
from datasets.paths import MOUSEACH_DATASET_PATH, MOUSE_DATASET_PATH


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


def get_signal_histograms(dataset, path, nr_of_bins, separated_histograms=False):
    conditions = 3
    channels = 10
    trials = 20

    for cond in range(conditions):
        for trial in range(trials):
            signals = []

            if separated_histograms:
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(111, projection='3d')

            for channel in range(channels):
                signal = dataset.signal[cond, trial, channel]
                if separated_histograms:
                    hist_values, bins = np.histogram(signal, nr_of_bins)
                    xs = (bins[:-1] + bins[1:]) / 2
                    ax.bar(xs, hist_values, width=bins[1] - bins[0], zs=channel * 5, zdir='y', alpha=0.8)
                else:
                    signals.append(signal)

            if not separated_histograms:
                signals = np.concatenate(signals).ravel()
                plt.hist(signals, nr_of_bins)

            plot_title = "Cond:{}_Trial:{}".format(cond, trial)
            plot_save_path = "{}/Channels_together/Cond:{}_Trial:{}_Multiple:{}".format(path, cond, trial,
                                                                                        separated_histograms)
            show_and_plot(plot_save_path, plot_title, show=True)

    for cond in range(conditions):
        for channel in range(channels):
            signals = []

            if separated_histograms:
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(111, projection='3d')

            for trial in range(trials):
                signal = dataset.signal[cond, trial, channel]
                if separated_histograms:
                    hist, bins = np.histogram(signal, nr_of_bins)
                    xs = (bins[:-1] + bins[1:]) / 2
                    ax.bar(xs, hist, zs=channel * 3, zdir='y', alpha=0.8)
                else:
                    signals.append(signal)

            if not separated_histograms:
                signals = np.concatenate(signals).ravel()
                plt.hist(signals, nr_of_bins)

            plot_title = "Cond:{}_Channel:{}".format(cond, channel)
            plot_save_path = "{}/Trials_together/Cond:{}_Channel:{}_Multiple:{}".format(path, cond, channel,
                                                                                        separated_histograms)
            show_and_plot(plot_title, plot_save_path, show=True)

    for trial in range(trials):
        for channel in range(channels):
            signals = []

            if separated_histograms:
                fig = plt.figure(figsize=(16, 10))
                ax = fig.add_subplot(111, projection='3d')

            for condition in range(conditions):
                signal = dataset.signal[condition, trial, channel]
                if separated_histograms:
                    hist, bins = np.histogram(signal, nr_of_bins)
                    xs = (bins[:-1] + bins[1:]) / 2
                    ax.bar(xs, hist, zs=channel * 10, zdir='y', alpha=0.5)
                else:
                    signals.append(signal)

            if not separated_histograms:
                signals = np.concatenate(signals).ravel()
                plt.hist(signals, nr_of_bins)

            plot_title = "Trial:{}_Channel:{}".format(trial, channel)
            plot_save_path = "{}/Conditions_together/Trial:{}_Channel:{}_Multiple:{}".format(path, trial, channel,
                                                                                             separated_histograms)
            show_and_plot(plot_title, plot_save_path, show=True)


if __name__ == '__main__':
    dataset = CatLFPStimuli()

    # dataset = MouseLFP(MOUSE_DATASET_PATH, cutoff_freq=[1, 80], channels_to_keep=[-1],
    #                    conditions_to_keep=[1], trials_to_keep=[0], normalization="Zsc")
    get_signal_histograms(dataset, "/home/pasca/School/Licenta/Naturix/Histograms/CatLFP", nr_of_bins=256,
                          separated_histograms=True)
