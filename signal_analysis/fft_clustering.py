from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from sklearn.cluster import AgglomerativeClustering

from datasets.CatLFPStimuli import CatLFPStimuli
from signal_analysis.heatmap import heatmap_from


def get_spectra(overlap, fourier_window_size, desired_slice_length, movies_to_keep, show=False):
    nr_trials = 20
    nr_channels = 20
    channels = [5]

    dataset = CatLFPStimuli(movies_to_keep=movies_to_keep, cutoff_freq=None, normalization='std')
    signals = dataset.signal

    spectrums = []

    for movie in movies_to_keep:
        for trial in range(nr_trials):
            for channel in channels:
                example = signals[movie, trial, channel, 150:]
                freqs, times, spectrum = spectrogram(example, 1000, window='blackman', nperseg=fourier_window_size,
                                                     noverlap=overlap,
                                                     detrend=False,
                                                     nfft=10 * fourier_window_size)
                spectrum = spectrum[:300, :]
                if show:
                    plt.pcolormesh(times, freqs[:300], 10 * np.log10(spectrum + 1), cmap='viridis')
                    plt.colorbar()
                    plt.show()
                slice_length = int(desired_slice_length / (times[1] - times[0]))
                for i in range(0, len(times) - slice_length, slice_length):
                    dicty = {"spectrum": spectrum[:, i:i + slice_length],
                             "movie": movie,
                             "trial": trial,
                             "channel": channel,
                             "slice": "{}-{}".format(times[i], times[i + slice_length])}
                    spectrums.append(dicty)
    return spectrums


def cluster(spectrums):
    linkages = ["“ward”, “complete”, “average”, “single”"]
    X = np.zeros((len(spectrums), spectrums[0]["spectrum"].size))
    for i, sample in enumerate(spectrums):
        X[i] = sample["spectrum"].flatten()
    clustering_labels = AgglomerativeClustering(affinity='l1', linkage="complete").fit_predict(X)
    n, bins, _ = plt.hist(clustering_labels, )
    plt.show()


def dist_matrix(spectrums, methods, movies_to_keep):
    spectrum_npy = np.stack([sample["spectrum"].flatten() for sample in spectrums])
    print(spectrum_npy.shape)
    for method in methods:
        results = heatmap_from(spectrum_npy, 'correlation', methods=[method],
                               save_location="./FFT_Correlations/",
                               plot_title="Cat_1s_windows-Movies:{}".format(movies_to_keep), show=True)

        cluster_start = int(input("Give cluster start"))
        cluster_stop = int(input("Give cluster end"))

        plt.figure(figsize=(20, 10))
        relevant_clusters = results[0][1][cluster_start:cluster_stop]
        relevant_clusters_slices = [spectrums[index]["slice"] for index in relevant_clusters]
        relevant_slice_counter = Counter(relevant_clusters_slices)
        print(relevant_slice_counter)
        plt.bar(relevant_slice_counter.keys(), relevant_slice_counter.values())
        plt.title("Slices histogram")
        plt.show()

        relevant_clusters_trials = [spectrums[index]["trial"] for index in relevant_clusters]
        plt.hist(relevant_clusters_trials, bins=range(20))
        plt.title("Trials histogram")
        plt.show()

        relevant_clusters_trials = [spectrums[index]["movie"] for index in relevant_clusters]
        plt.hist(relevant_clusters_trials)
        plt.title("Movie histogram")
        plt.show()

        relevant_clusters_channels = [spectrums[index]["channel"] for index in relevant_clusters]
        plt.hist(relevant_clusters_channels, bins=range(47))
        plt.title("Channels histogram")
        plt.show()


if __name__ == '__main__':
    fourier_window_size = 500
    sliding_window = 250
    movies_to_keep = [0, 1, 2]
    overlap = fourier_window_size - sliding_window
    spectrums = get_spectra(overlap, fourier_window_size, movies_to_keep=movies_to_keep, desired_slice_length=2, )

    dist_matrix(spectrums, methods=["average"], movies_to_keep=movies_to_keep)
    # cluster(spectrums)
