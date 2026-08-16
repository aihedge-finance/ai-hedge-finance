from scipy.signal import butter, sosfiltfilt


def sos_butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = sos_butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data)

    return y
