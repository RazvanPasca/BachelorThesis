import numpy as np
from scipy.signal import butter, filtfilt


def rescale(x, old_max, old_min, new_max, new_min):
    val = (new_max - new_min) * (x - old_min) / (old_max - old_min) + new_min
    return val


def moving_average_smooth(x, nr_points):
    window = np.ones(nr_points) / nr_points
    x_smooth = np.convolve(x, window, mode="same")
    return x_smooth


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False, output="ba")
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y
