import inout
import torch
import plotter
from generator import Generator
from discriminator import Discriminator, WGANDiscriminator
from typing import Dict
import numpy as np
import util

def latent_vector_from_numpy(batch_size, latent_size):
    return torch.from_numpy((np.random.rand(batch_size, latent_size)-0.5)*10000).float()

def latent_vector_uniform_batch(batch_size, latent_size):
    return torch.rand(batch_size, latent_size)

def latent_vector_uniform(size):
    """
    Returns latent vector sampled from a uniform distribution.
    """
    #TODO: Check if values in [-1,1] and if [-1,1] is the only adequate window
    return torch.rand(size)

def latent_vector_normal(batch_size, latent_size):
    """
    Returns latent vector sampled from the (standard) normal distribution.
    """
    #TODO: Check if values in [-1,1]
    return torch.randn(batch_size, latent_size, 1)

def load_dataset(path):
    raise NotImplementedError

def sampleZ(batch_size, latent_size, latent_dist):
    if latent_dist == 'normal':
        return latent_vector_normal(batch_size, latent_size)
    elif latent_dist == 'uniform':
        return latent_vector_uniform(batch_size, latent_size)


def train(args : Dict):
    #Initialize parameters
    num_iterations = args.get('num_iterations')
    k = args.get('k')
    batch_size = args.get('batch_size')
    latent_size = args.get('latent_size')
    dataset_path = args.get('dataset_path')
    dataset = inout.AudioDataset(dataset_path.split('/')[-1], dataset_path)
    use_wgan = args.get('use_wgan')
    latent_dist = args.get('latent_dist')
    learning_rate = args.get('learning_rate')


    #Initialize Generator and Discriminator
    G = Generator(latent_size, model_size=1)
    D = Discriminator(model_size=1)

    #Use binary cross entropy as loss function
    loss_fn = torch.nn.BCELoss()

    optimizerD = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=learning_rate*10, betas=(0.5, 0.999))

    #optimizerD = torch.optim.RMSprop(D.parameters(), lr=learning_rate)
    #optimizerG = torch.optim.RMSprop(G.parameters(), lr=learning_rate)

    lossDs = []
    lossGs = []
    epoch = 0
    epoch_iteration = 0
    for i in range(num_iterations):
        #Check if epoch is finished and enter new epoch
        if epoch_iteration >= dataset.epoch_length(batch_size)-1:
            dataset.shuffle()
            epoch = epoch+1
            epoch_iteration = 0

        #Train Discriminator k times
        for j in range(k):
            """#Skip discriminator training when getting to strong
            if i>=500 and lossDs[len(lossDs)-1] < lossGs[len(lossGs)-1]:
                lossDs.append(lossDs[len(lossDs)-1])
                continue"""

            #Sample minibatch of m noise samples to train discriminator
            z = torch.unsqueeze(latent_vector_from_numpy(batch_size, latent_size), 0)
            z = sampleZ(batch_size, latent_size, latent_dist)
            y = G(z)

            #Sample minibatch of m examples from data generating distribution
            batch = dataset.get_batch(epoch_iteration, batch_size, epoch)
            x = torch.unsqueeze((torch.from_numpy(batch)), 1).float()


            #Start: Update Discriminator weights
            optimizerD.zero_grad()
            #Compute losses
            if not use_wgan:

                errDreal = loss_fn(D(x).view(-1), torch.full((batch_size,), 1).float())
                errDfake = loss_fn(D(y).view(-1), torch.full((batch_size,), 0).float())
                errDreal.backward()
                errD = errDreal + errDfake
                lossDs.append(errD.item())
                errDfake.backward()
            else:
                errDreal = D(x)
                errDreal.backward(torch.FloatTensor([1]))
                errDfake = D(y)
                errDfake.backward(torch.FloatTensor([1])*-1)
                errD = errDreal+errDfake
                lossDs.append(errD.item())
            optimizerD.step()
            #End: Update Discriminator weights

        #Train Generator

        #Sample minibatch of m noise samples to train generator
        z = sampleZ(batch_size, latent_size, latent_dist)

        optimizerG.zero_grad()
        if not use_wgan:
            errG = loss_fn(D(G(z)).view(-1), torch.full((batch_size,), 1).float())
            errG.backward()
        else:
            errG = D(G(z))
            errG.backward(torch.FloatTensor([1]))
        lossGs.append(errG.item())
        optimizerG.step()

        #Generate samples, test discriminator and plot losses every (?) iterations to check progress:
        if i % 1000 == 0:
            generate_audio(G, 'it'+str(i)+'_', 3, batch_size, latent_size, latent_dist)
            test_discriminator(D, G, batch_size, dataset, latent_size, epoch, epoch_iteration, latent_dist)
            print(lossDs)
            print(lossGs)
            plotter.plot_losses(lossDs, lossGs)
        i+=1
        epoch_iteration+=1
    return D, G

def generate_audio(G : Generator, prefix : str, amount, batch_size, latent_size, latent_dist):
    x = 0
    while x < amount:
        # Generate and save sounds with G
        z = sampleZ(batch_size, latent_size, latent_dist)
        amplitudes = G(z).detach().numpy()
        for i in range(batch_size):
            inout.write_wav_file(args.get('generate_path')+'/'+prefix+str(x), amplitudes[i, 0, :])
            plotter.plot_wav(args.get('generate_path')+'/'+prefix+str(x)+'.wav')
            x+=1

def test_discriminator(D : Discriminator, G : Generator, batch_size, dataset : inout.AudioDataset, latent_size, epoch, epoch_it, latent_dist):
    batch = dataset.get_batch(epoch_it, batch_size, epoch)
    plotter.plot_1d_array(batch[0])
    x = torch.unsqueeze((torch.from_numpy(batch)), 1).float()
    print('**** Real Data ****')
    print(D(x))
    z = sampleZ(batch_size, latent_size, latent_dist)
    print('**** Fake Data ****')
    print(D(G(z)))


if __name__ == '__main__':
    args = {'num_iterations': 10000000, 'k': 1, 'batch_size': 4, 'latent_size': 100,
            'dataset_path': './datasets/nsynth/nsynth-test/4keys', 'learning_rate': 0.0002, 'generate_path': './generated_sounds',
            'use_wgan': False, 'latent_dist': 'normal'}
    D, G = train(args)


    """#Generate and save sounds with G
    z = torch.unsqueeze(latent_vector_uniform_batch(args.get('batch_size'), args.get('latent_size')), 0)
    amplitudes = G(z).detach().numpy()
    for x in range(25):
        for i in range(args.get('batch_size')):
            index = x*args.get('batch_size')+i
            inout.write_wav_file(args.get('generate_path')+'/'+str(x), amplitudes[0, i, :])"""

    """G = Generator()
    z = latent_vector_uniform_batch(3, 100)
    z = torch.unsqueeze(z, 0)#torch.ones((1, 1, 100))
    #z = torch.unsqueeze(z, 1)
    z = G(z)
    z = z.detach().numpy()
    print(z)
    #inout.write_wav_file('generated', y[0, 0, :], samplerate=16000)
    #plotter.plot_wav('generated.wav')


    D = Discriminator()
    #s, x = inout.load_wav_file('testwav.wav')
    x = z
    print(x)
    x = torch.from_numpy(x)
    x = torch.tensor(x, dtype=torch.float32)
   #x = torch.unsqueeze(x, 0)
    #x = torch.unsqueeze(x, 1)
    x = D(x)
    print(x)"""
