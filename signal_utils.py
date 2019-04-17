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


def mu_law_fn(x, mu=256):
    """Maps [-1,1] to [-1,-1] with a readjusted distribution"""
    val = np.sign(x) * (np.log(1 + (mu - 1) * np.absolute(x)) / np.log(1 + mu - 1))
    assert (-1 <= np.min(val))
    assert (np.max(val) <= 1)
    return val


def mu_law_encoding(x, mu=256):
    bin = np.rint(rescale(x, 1, -1, mu - 1, 0))
    return bin


def inv_mu_law_fn(x, mu=256):
    """Maps [-1,1] decoded from bin to [-1,1] initial rescale of the signals
    To get the original values between -300 and 400 for example, we need to use rescale fn above with each channels'
    min and max values"""
    assert (-1 <= x <= 1)
    val = np.sign(x) * (1 / (mu - 1)) * (((1 + mu - 1) ** np.abs(x)) - 1)
    assert (-1 <= val <= 1)
    return val
