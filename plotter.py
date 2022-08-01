import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

import loader


def plot_wav(path, size=None):
    """
    Plots a WAV-file with time on the x-axis and amplitude on the y-axis.

    :param path: Path to the WAV-file
    :param size: How many samples to plot, default is the whole file
    """
    samplerate, amplitude = wf.read(path)

    if size is None:
        x = range(amplitude.size)
    else:
        x = range(size)
        amplitude = amplitude[0:size]
    plt.plot(x, amplitude)
    plt.show()