import math

import numpy as np
import scipy.io.wavfile as wf
import torch

import inout
import plotter
import main
import util
from generator import GeneratorWaveGAN
from discriminator import Discriminator
from discriminator import DiscriminatorWaveGAN



def run_tests():

    D = DiscriminatorWaveGAN(model_size=1)


    G = GeneratorWaveGAN(latent_size= 100, model_size=1)
    x = torch.randn(1, 100, 1)
    print(D(G(x)))
#    plotter.plot_dataset('audioComplete', './datasets/nsynth/nsynth-test/audioComplete', 64000)
    """"
    
    z = torch.unsqueeze(inout.latent_vector_from_numpy(4, 100), 1)
    print(z.size())
    G = Generator(100, 1)
    print(G)
    z = main.sampleZ(1, 10, 'normal')
    z = torch.randn(1, 100, 1)
    D = Discriminator(1)
    print(D)
    D(G(z))dataset = inout.AudioDataset()
    a, sound = inout.load_wav_file('testwav.wav')
    for i in range(64000):
        if (i%100)==50:
            x = 25
        if (i%100)==0:
            x = -25
        sound[i]=x


    print(sound)
    inout.write_wav_file('written_wav', sound, samplerate=16000)

    #print(main.latent_vector_from_numpy(4, 4))

   # plotter.plot_wav('written_wav.wav', size=100)

    s, a = inout.load_wav_file('./generated_sounds/it4000_2.wav')
    plotter.plot_wav('./datasets/nsynth/nsynth-test/4keys/keyboard_acoustic_004-090-075.wav')
    print(min(a))
    print(sound.shape)
    print(a.shape)
    #inout.write_wav_file('written_wav', a)

    ### Test random sampling ###

    #print(main.latent_vector_uniform(50))
    #print(main.latent_vector_normal(50))

    ### Test plot_wav function ###
    #plotter.plot_wav('./testwav.wav')

    #plotter.plot_wav('./testwav.wav', size=100)

    ### Test what kind of data is returned by scipys wavfile.read() ###
    #a, b = wf.read('./testwav.wav')
    #print(a)
    #print("####")
    #print(b)
    """""

if __name__ == '__main__':
    run_tests()