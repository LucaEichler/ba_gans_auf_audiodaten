import random
import numpy as np

def random_permutation(length):
    x = np.arange(length)
    x = np.random.permutation(x)
    return x

def write_numpy_file(filepath, array):
    np.save(filepath, array)

def load_numpy_file(filepath):
    return np.load(filepath)

def shift_above_zero(array):
    x = np.ones(array.shape)
    x *= np.abs(np.min(array))
    array = np.add(array, x)
    return array

