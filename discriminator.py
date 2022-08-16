import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, model_size):
        super(Discriminator, self).__init__()

        self.model_size = model_size
        self.conv1 = nn.Conv1d(1, model_size, kernel_size=25, stride=3, padding=0, bias=False)

        self.conv2 = nn.Conv1d(model_size, model_size*2, kernel_size=21, stride=5, padding=0, bias=False)

        self.conv3 = nn.Conv1d(model_size*2, model_size*4, kernel_size=17, stride=3, padding=0,
                               bias=False)

        self.conv4 = nn.Conv1d(model_size*4, model_size * 8, kernel_size=21, stride=5, padding=0,
                               bias=False)

        self.conv5 = nn.Conv1d(model_size * 8, model_size * 16, kernel_size=15, stride=5, padding=0,
                               bias=False)

        self.conv6 = nn.Conv1d(model_size * 16, model_size * 32, kernel_size=9, stride=3, padding=0,
                               bias=False)

        self.conv7 = nn.Conv1d(model_size * 512, 1, kernel_size=1, stride=4, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        """"self.conv1 =nn.Conv1d(in_channels=batch_size, out_channels=batch_size, kernel_size=4, stride=4)
        self.conv2 =nn.Conv1d(in_channels=batch_size, out_channels=batch_size, kernel_size=4, stride=4)
        self.conv3 =nn.Conv1d(in_channels=batch_size, out_channels=batch_size, kernel_size=4, stride=4)
        self.conv4 =nn.Conv1d(in_channels=batch_size, out_channels=batch_size, kernel_size=4, stride=4)
        self.lrelu1 =nn.LeakyReLU(negative_slope=0.2)
        self.lrelu2 =nn.LeakyReLU()
        self.lrelu3 =nn.LeakyReLU()
        self.lrelu4 =nn.LeakyReLU()
        self.fc1 = nn.Linear(in_features=250, out_features=1)
        self.sigmoid = nn.Sigmoid()"""

    def run(self, x):
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = x.view(x.size(dim=0), 512, self.model_size)
        x = self.lrelu(self.conv7(x))
        return x

    def forward(self, x):
        x = self.run(x)
        x = self.sigmoid(x)
        return x


class WGANDiscriminator(Discriminator):
    def __init__(self, model_size):
        super(WGANDiscriminator, self).__init__(model_size)


    def forward(self, x):
        x = super().run(x)
        x = x.mean(0)
        x = x.view(1)
        return x