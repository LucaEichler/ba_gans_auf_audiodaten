from torch import autograd
from torch.autograd import Variable

import inout
import torch
import plotter
from generator import Generator, GeneratorWaveGAN
from discriminator import Discriminator, WGANDiscriminator
from typing import Dict
import numpy as np


#Taken from https://github.com/Zeleni9/pytorch-wgan/ TODO: Write own version.
def calculate_gradient_penalty(batch_size, real_sounds, fake_sounds, D, G):
    lambda_term = 10

    eta = torch.FloatTensor(batch_size, 1,1).uniform_(0,1)
    eta = eta.expand(batch_size, real_sounds.size(1), real_sounds.size(2))

    interpolated = eta * real_sounds + ((1 - eta) * fake_sounds)


    # define it to calculate gradient
    interpolated = Variable(interpolated, requires_grad=True)

    # calculate probability of interpolated examples
    prob_interpolated = D(interpolated)

    # calculate gradients of probabilities with respect to examples
    gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(
                               prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    return grad_penalty

def latent_vector_from_numpy(batch_size, latent_size):
    return torch.from_numpy((np.random.rand(batch_size, latent_size)-0.5)*10000).float()

def latent_vector_uniform_batch(batch_size, latent_size):
    return torch.rand(batch_size, latent_size)

def latent_vector_uniform(batch_size, latent_size):
    """
    Returns latent vector sampled from a uniform distribution.
    """
    #TODO: Check if values in [-1,1] and if [-1,1] is the only adequate window
    return torch.rand(batch_size, latent_size, 1)

def latent_vector_normal(batch_size, latent_size):
    """
    Returns latent vector sampled from the (standard) normal distribution.
    """
    #TODO: Check if values in [-1,1]
    return torch.randn(batch_size, latent_size)

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
    mode = args.get('mode')
    latent_dist = args.get('latent_dist')
    learning_rate = args.get('learning_rate')


    #Initialize Generator and Discriminator
    G = GeneratorWaveGAN(latent_size, model_size=1)
    D = Discriminator(model_size=1)
    if mode == 'wgan' or mode == 'wgan-gp':
        D = WGANDiscriminator(model_size=1)

    #Use binary cross entropy as loss function
    loss_fn = torch.nn.BCELoss()

    torch.autograd.set_detect_anomaly(True)
    optimizerD = torch.optim.Adam(D.parameters(), lr=learning_rate*0.1, betas=(0.5, 0.999))
    optimizerG = torch.optim.Adam(G.parameters(), lr=learning_rate*0.5, betas=(0.5, 0.999))
    if mode == 'wgan':
        optimizerD = torch.optim.RMSprop(D.parameters(), lr=0.00005)
        optimizerG = torch.optim.RMSprop(G.parameters(), lr=0.00005)

    lossDs = []
    lossGs = []
    epoch = 0
    epoch_iteration = 0
    for i in range(num_iterations):

        #
        for p in D.parameters():
            p.requires_grad = True

        #Check if epoch is finished and enter new epoch
        if epoch_iteration >= dataset.epoch_length(batch_size)-1:
            dataset.shuffle()
            epoch = epoch+1
            epoch_iteration = 0

        lossDs_temp = []
        #Train Discriminator k times
        for j in range(k):

            """#Skip discriminator training when getting to strong
            if i>=500 and lossDs[len(lossDs)-1] < lossGs[len(lossGs)-1]:
                lossDs.append(lossDs[len(lossDs)-1])
                continue"""

            #Sample minibatch of m noise samples to train discriminator
            z = torch.unsqueeze(latent_vector_from_numpy(batch_size, latent_size), 0)
            z = sampleZ(batch_size, latent_size, latent_dist)
            print(z.size())
            y = G(z)

            #Sample minibatch of m examples from data generating distribution
            batch = dataset.get_batch(epoch_iteration, batch_size, epoch, output_size=65536)
            x = torch.unsqueeze((torch.from_numpy(batch)), 1).float()

            #Start: Update Discriminator weights
            optimizerD.zero_grad()
            D.zero_grad()

            #Compute losses
            if not (mode == 'wgan' or mode == 'wgan-gp'):

                errDreal = loss_fn(D(x).view(-1), torch.full((batch_size,), 1).float())
                errDfake = loss_fn(D(y).view(-1), torch.full((batch_size,), 0).float())

                errDreal.backward()
                errDfake.backward()

                errD = errDreal + errDfake
            else:
                errDreal = D(x)
                errDreal.backward(torch.FloatTensor([1]))
                errDfake = D(y)
                errDfake.backward(torch.FloatTensor([1])*-1)
                errD = errDfake-errDreal
                wasserstein_distance = errDreal-errDfake
                if mode == 'wgan-gp':
                    gradient_penalty = calculate_gradient_penalty(batch_size, x.data, y.data, D, G)
                    gradient_penalty.backward()
                if mode == 'wgan':
                    #Weight clipping - before or after computing Gradients?
                    weight_clipping_limit = 0.01
                    for p in D.parameters():
                        p.data.clamp_(-weight_clipping_limit, weight_clipping_limit)
            optimizerD.step()
            lossDs_temp.append(errD.item())
        lossDs.append(sum(lossDs_temp)/k)
        #End: Update Discriminator weights

        for p in D.parameters():
            p.requires_grad = False


        #Train Generator
        optimizerG.zero_grad()
        G.zero_grad()

        #Sample minibatch of m noise samples to train generator
        z = sampleZ(batch_size, latent_size, latent_dist)

        if not (mode == 'wgan' or mode == 'wgan-gp'):
            errG = loss_fn(D(G(z)).view(-1), torch.full((batch_size,), 1).float())
            errG.backward()
        else:
            errG = D(G(z))
            errG.backward(torch.FloatTensor([1]))
        lossGs.append(errG.item())
        optimizerG.step()

        #Generate samples, test discriminator and plot losses every (?) iterations to check progress:
        if i % 1000 == 0:
            generate_audio(G, f'it{i}_', 3, batch_size, latent_size, latent_dist)
            test_discriminator(D, G, batch_size, dataset, latent_size, epoch, epoch_iteration, latent_dist)
            plotter.plot_losses(lossDs, lossGs)
            #for p in G.parameters():
                #print(p.data)

        i+=1
        epoch_iteration+=1

        if i % 10000 == 0:
            test_discriminator(D, G, dataset.__len__(), dataset, latent_size, 0, 0, latent_dist)
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
    batch = dataset.get_batch(epoch_it, batch_size, epoch, output_size=65536)
    plotter.plot_1d_array(batch[0])
    x = torch.unsqueeze((torch.from_numpy(batch)), 1).float()
    print('**** Real Data ****')
    print(D(x))
    z = sampleZ(batch_size, latent_size, latent_dist)
    print('**** Fake Data ****')
    print(D(G(z)))


if __name__ == '__main__':
    args = {'num_iterations': 10000000, 'k': 1, 'batch_size': 1, 'latent_size': 100,
            'dataset_path': './datasets/nsynth-test/keyboard_accoustic', 'learning_rate': 0.0002, 'generate_path': './generated_sounds',
            'mode': 'dcgan', 'latent_dist': 'normal'}
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
