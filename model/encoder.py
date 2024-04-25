# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
from torch import nn
from torchvision.models import resnet18

class ImageEncoder(torch.nn.Module):

    def __init__(self, latent_dim):
        super(ImageEncoder, self).__init__()

        self.latent_dim = latent_dim
        self.base_architecture= 'resnet18'
        self.width = 128

        self.base_model = resnet18(pretrained=True)
        self.feat_layers= list(self.base_model.children())[:-1] # remove the last linear layer of 512 -> 1000
        self.feat_net= nn.Sequential(*self.feat_layers)

        self.fc_layers= [
                    nn.Linear(512, self.width),
                    nn.LeakyReLU(),
                    nn.Linear(self.width, self.latent_dim),
                ]

        self.fc_net = nn.Sequential(*self.fc_layers)

    def forward(self, x):
        '''
        x : [ B, C, H, W ] 
        '''
        x= self.feat_net(x)                 # [ B, 512, 1, 1 ]
        x= x.view(x.shape[0], x.shape[1])   # [ B, 512 ]
        x= self.fc_net(x)                   # [ B, latent_dim ]
        return x