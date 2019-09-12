import torch 
import torch.nn as nn
import torch.optim as optim 

import numpy as np
from torch.autograd import Variable
from torch.autograd import grad

class Generator(nn.Module):
    def __init__(self, latent_size=20):

        super(self.__class__, self).__init__()
        #initilization of variables
        self.latent_size = latent_size

        self.preprocess = nn.Sequential(
            nn.Linear(latent_size, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(500, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        )

        self.Tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, z):
        h = self.preprocess(z)
        h = h[:,:,None,None]
        x = self.decoder(h)
        x = x.view(-1, 3, 32, 32)
        x = self.Tanh(x)
        return x