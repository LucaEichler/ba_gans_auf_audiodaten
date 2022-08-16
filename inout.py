import os
import numpy as np
import scipy.io.wavfile as wf
import torch
import torch.utils.data as data

import util


def latent_vector_from_numpy(batch_size, latent_size):
    return torch.from_numpy((np.random.rand(batch_size, latent_size)-0.5)*10000).float()

class AudioDataset(data.Dataset):

    def save_dataset(self, filepath):
        util.write_numpy_file(filepath, self.data)

    def load_dataset(self, filepath):
        self.data = util.load_numpy_file(filepath)

    def __init__(self, dataset_name : str,dataset_path, data_size=64000):
        super(AudioDataset, self)
        self.data_size = data_size
        try:
            self.load_dataset('./datasets/numpy/'+dataset_name+'.npy')
        except:
            working_dir = os.listdir(dataset_path)
            self.data = np.zeros((len(working_dir),data_size))
            i = 0
            for file in working_dir:
                try:
                    s, a = load_wav_file(dataset_path+'/'+file)
                    #x = self.data.shape[0]+1
                    #self.data.resize((x, data_size), refcheck=False)
                    self.data[i, :] = a
                    print(i)
                    i+=1
                except FileNotFoundError:
                    #Array um 1 verringern
                    x = self.data.shape[0]-1
                    self.data.resize((x, data_size), refcheck=False)
                    print('File '+file+' was not found.')
            self.save_dataset('./datasets/numpy/'+dataset_name+'.npy')
        self.max_val = max(self.data.max(), -self.data.min())
        self.current_permutation = np.arange(self.data.shape[0])

    def __getitem__(self, index):
        return self.data[index, :]

    def __len__(self):
        return self.data.shape[0]

    def shuffle(self):
        self.current_permutation = util.random_permutation(self.__len__())

    def get_batch(self, epoch_iteration, batch_size, epoch):
        #epoch_iteration = iteration-epoch*self.epoch_length(batch_size)
        start_index = epoch_iteration * batch_size
        end_index = start_index + batch_size
        if end_index > self.__len__():
            raise IndexError
        else:
            batch = np.zeros((batch_size, self.data_size))
            for i in range(batch_size):
                #Apply random permutation each epoch
                x = start_index+i
                batch[i, :]=self.__getitem__(self.current_permutation[int(x)])
                i+=1
        #Normalize
        batch = batch/self.max_val
        """noise_array = np.random.rand(batch.shape[0], batch.shape[1])
        noise_array-=0.5
        noise_array*=2
        batch = np.add(batch, noise_array)
        batch = batch/np.max(batch)"""
        #Make values positive
        batch = util.shift_above_zero(batch)
        return batch


    def epoch_length(self, batch_size):
        """
        Computes how many iterations one epoch lasts for the given batch_size.
        """
        return self.__len__()/batch_size

    def get_data(self):
        return self.data

"""class AudioCompleteSet(AudioDataset, ):
    def __init__(self, data_path):
        super(AudioDataset, self).__init__(dataset_name='audioComplete', dataset_path = data_path, data_size=64000)
"""

def load_wav_file(path):
    samplerate, amplitude = wf.read(path)
    return samplerate, amplitude

def write_wav_file_to_path(filename, samplerate, amplitude):
        wf.write(filename, samplerate, amplitude)


def write_wav_file(filename, amplitude, samplerate=16000):
    write_wav_file_to_path(filename+'.wav', samplerate, amplitude)

