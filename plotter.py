import matplotlib.pyplot as plt
import scipy.io.wavfile as wf

import inout


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

def plot_losses(lossDs, lossGs):
    iterations = range(len(lossDs))
    plt.plot(iterations, lossDs, color='g')
    plt.plot(iterations, lossGs, color ='r')
    plt.show()

def plot_1d_array(values):
    x = range(len(values))
    plt.plot(x, values, color='blue')
    plt.show()

def plot_1d_arrrays(values):
    for x in range(len(values)):
        plot_1d_array(values[x, :])

def plot_dataset(name, path, size):
    dataset = inout.AudioDataset(dataset_name=name, dataset_path=path, data_size=size)
    data = dataset.get_data()
    plot_1d_arrrays(data)


