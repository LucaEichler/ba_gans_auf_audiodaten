import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F



class GeneratorWaveGAN(nn.Module):

    def __init__(self, latent_size, model_size):
        super(GeneratorWaveGAN, self).__init__()
        self.model_size = model_size

        #TODO: Bias or no Bias?

        #self.tconv1 = nn.ConvTranspose1d(latent_size, model_size*512, kernel_size=1, stride = 4, padding = 0, bias = False)
        self.fc1 = nn.Linear(latent_size, model_size*512)

        self.tconv2 = nn.ConvTranspose1d(model_size*32, model_size*16, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tconv3 = nn.ConvTranspose1d(model_size*16, model_size*8, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tconv4 = nn.ConvTranspose1d(model_size*8, model_size*4, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tconv5 = nn.ConvTranspose1d(model_size*4, model_size*2, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tconv6 = nn.ConvTranspose1d(model_size*2, model_size, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tconv7 = nn.ConvTranspose1d(model_size, 1, kernel_size=25, stride = 4, padding = 11, output_padding=1, bias = False)

        self.tanh = nn.Tanh()

        # Input is the latent vector Z.


    def forward(self, x):
        print(x.size())
        x = self.fc1(x)
        print(x.size())
        x = x.view(-1, self.model_size*32, 16)
        x = x = torch.relu(x)
        print(x.size())

        x = torch.relu(self.tconv2(x))
        print(x.size())

        x = torch.relu(self.tconv3(x))
        print(x.size())
        print(f"{x.size()=}")


        x = torch.relu(self.tconv4(x))
        print(x.size())


        x = torch.relu(self.tconv5(x))
        print(x.size())


        x = torch.relu(self.tconv6(x))
        print(x.size())


        x = self.tconv7(x)
        print(x.size())


        x = self.tanh(x)
        return x


class Generator(nn.Module):

    def __init__(self, latent_size, model_size):
        super(Generator, self).__init__()
        self.model_size = model_size

        #TODO: Bias or no Bias?


        self.tconv1 = nn.ConvTranspose1d(latent_size, model_size*512, kernel_size=1, stride = 4, padding = 0, bias = False)

        self.tconv2 = nn.ConvTranspose1d(model_size*32, model_size*16, kernel_size=9, stride = 3, padding = 0, bias = False)

        self.tconv3 = nn.ConvTranspose1d(model_size*16, model_size*8, kernel_size=15, stride = 5, padding = 0, bias = False)

        self.tconv4 = nn.ConvTranspose1d(model_size*8, model_size*4, kernel_size=21, stride = 5, padding = 0, bias = False)

        self.tconv5 = nn.ConvTranspose1d(model_size*4, model_size*2, kernel_size=17, stride = 3, padding = 0, bias = False)

        self.tconv6 = nn.ConvTranspose1d(model_size*2, model_size, kernel_size=21, stride = 5, padding = 0, bias = False)


        self.tconv7 = nn.ConvTranspose1d(model_size, 1, kernel_size=25, stride = 3, padding = 0, bias = False)
        self.tanh = nn.Tanh()

        """self.tconv1 = nn.ConvTranspose2d(nz, ngf * 8,
                                         kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)

        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4,
                                         4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)

        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2,
                                         4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)

        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(ngf * 2, ngf,
                                         4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(ngf, nc,
                                         4, 2, 1, bias=False)
        # Output Dimension: (nc) x 64 x 64

    def forward(self, x):
        x = F.relu(self.bn1(self.tconv1(x)))
        print(x.size())
        x = F.relu(self.bn2(self.tconv2(x)))
        print(x.size())
        x = F.relu(self.bn3(self.tconv3(x)))
        print(x.size())
        x = F.relu(self.bn4(self.tconv4(x)))
        x = torch.tanh(self.tconv5(x))

        return x"""

    def forward(self, x):
        x = torch.relu(self.tconv1(x))
        x = x.view(x.size(dim=0), self.model_size*32, 16)
        x = torch.relu(self.tconv2(x))
        x = torch.relu(self.tconv3(x))
        x = torch.relu(self.tconv4(x))
        x = torch.relu(self.tconv5(x))
        x = torch.relu(self.tconv6(x))
        x = self.tconv7(x)
        x = self.tanh(x)
        return x
"""
        self.tconv1 = nn.ConvTranspose1d(latent_size, 250, kernel_size=1)
        self.tconv1 = nn.ConvTranspose1d(250, , kernel_size=4)

        self.tconv1 = nn.ConvTranspose1d(8, 1, kernel_size=4)

        self.tconv1 = nn.ConvTranspose1d(8, 1, kernel_size=4)

        self.tconv1 = nn.ConvTranspose2d(2, 10, kernel_size=4)


        self.tconv1 = nn.ConvTranspose1d(in_channels=batch, out_channels=batch, kernel_size=4, stride=4)
        self.tconv2 = nn.ConvTranspose1d(in_channels=batch, out_channels=batch, kernel_size=10, stride=10)
        self.tconv3 = nn.ConvTranspose1d(in_channels=batch, out_channels=batch, kernel_size=4, stride=4)
        self.tconv4 = nn.ConvTranspose1d(in_channels=batch, out_channels=batch, kernel_size=4, stride=4)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.tconv1(x)
        x = self.relu(self.tconv2(x))
        x = self.relu(self.tconv3(x))
        x = self.tanh(self.tconv4(x))
        #x = self.sigmoid(self.tconv4(x))

        return x"""
