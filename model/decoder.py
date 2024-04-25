# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn

class ImageDecoder(torch.nn.Module):

    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.width= 128
        self.nc= 3

        self.linear =  [
                    nn.Linear(self.latent_dim, self.width),
                    nn.LeakyReLU(),
                    nn.Linear(self.width, 1024),
                    nn.LeakyReLU(),
        ]
        self.linear= nn.Sequential(*self.linear)

        self.conv= [
                    nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d(32, self.nc, 4, stride=2, padding=1),
        ]
        self.conv= nn.Sequential(*self.conv)


    def forward(self, z):
        '''
        z : [ B, latent_dim ]
        '''
        x = self.linear(z)                  # [ B, 1024 ]  
        x = x.view(z.size(0), 64, 4, 4)     # [ B, 64, 4, 4 ]
        x = self.conv(x)                    # [ B, 3, 64, 64 ]
        return x