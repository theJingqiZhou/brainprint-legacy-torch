import numpy as np
from scipy import signal


def filter(data, low_freq, high_freq, sample_rate):
    b, a = signal.butter(4, [low_freq, high_freq], btype="bandpass", fs=sample_rate)
    filtered_signal = signal.lfilter(b, a, data, axis=1)

    return filtered_signal
