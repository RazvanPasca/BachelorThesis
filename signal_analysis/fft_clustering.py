import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram
from sklearn.cluster import AgglomerativeClustering

from datasets.CatLFPStimuli import CatLFPStimuli


def get_spectra(overlap, fourier_window_size, show=False):
    movies_to_keep = [0]
    nr_trials = 20
    nr_channels = 20

    dataset = CatLFPStimuli(movies_to_keep=movies_to_keep, cutoff_freq=None, normalization='std')
    signals = dataset.signal

    spectrums = []

    for movie in movies_to_keep:
        for trial in range(nr_trials):
            for channel in range(nr_channels):
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
                slice_length = int(1 / (times[1] - times[0]))
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


if __name__ == '__main__':
    fourier_window_size = 500
    sliding_window = 250
    overlap = fourier_window_size - sliding_window
    spectrums = get_spectra(overlap, fourier_window_size)
    cluster(spectrums)
