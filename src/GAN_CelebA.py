#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:06:11 2023

@author: frederikrygaard
"""

#%% Sources

#https://towardsdatascience.com/celebrity-face-generation-with-deep-convolutional-gans-40b96147a1c9

#%% Modules

from torch import (
    ones
    )

from torch.cuda import (
    is_available
    )

from torch.nn import (
    Module, 
    Linear,
    Conv2d,
    ConvTranspose2d,
    BatchNorm2d,
    Sequential,
    BCEWithLogitsLoss
    )

from torch.nn.functional import (
    leaky_relu,
    relu,
    tanh
    )

#%% Discriminator

class Discriminator(Module):
    def __init__(self,
                 conv_dim):
        super(Discriminator,self).__init__()
        
        self.conv_dim=conv_dim

        self.conv1=conv(3,conv_dim,4,batch_norm=False)
        self.conv2=conv(conv_dim,conv_dim*2,4,batch_norm=False)
        self.conv3=conv(conv_dim*2,conv_dim*4,4,batch_norm=False)
        self.conv4=conv(conv_dim*4,conv_dim*8,4,batch_norm=False)

        self.fc=Linear(conv_dim*8*2*2,1)


    def forward(self,x):
        x=F.leaky_relu(self.conv1(x),0.2)
        x=F.leaky_relu(self.conv2(x),0.2)
        x=F.leaky_relu(self.conv3(x),0.2)
        x=F.leaky_relu(self.conv4(x),0.2)
        x = x.view(-1, self.conv_dim*8*2*2)
        out=self.fc(x)
        return out
    