import loader
import torch


def latent_vector_uniform(size):
    """
    Returns latent vector sampled from a uniform distribution.
    """
    #TODO: Check if values in [-1,1] and if [-1,1] is the only adequate window
    return torch.rand(size)

def latent_vector_normal(size):
    """
    Returns latent vector sampled from the (standard) normal distribution.
    """
    #TODO: Check if values in [-1,1]
    return torch.randn(size)


if __name__ == '__main__':
    loader.load_wav_file()