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

from torch.optim import (
    Adam
    )

#%% Code

def conv(in_channels,out_channels,kernel_size,stride=2,padding=1,batch_norm=True):
    
    layers=[]
    conv=Conv2d(in_channels,out_channels,kernel_size,stride,padding,bias=False)

    layers.append(conv)

    if batch_norm:
        layers.append(BatchNorm2d(out_channels))

    return Sequential(*layers)


def deconv(in_channels,out_channels,kernel_size,stride=2,padding=1,batch_norm=True):
    layers=[]
    transpose_conv = ConvTranspose2d(in_channels,out_channels,kernel_size,
                                     stride,padding,bias=False)

    layers.append(transpose_conv)

    if batch_norm:
        layers.append(BatchNorm2d(out_channels))

    return Sequential(*layers)

class Discriminator(Module):
    def __init__(self,conv_dim):
        super(Discriminator,self).__init__()
        self.conv_dim=conv_dim

        self.conv1=conv(3,conv_dim,4,batch_norm=False)
        self.conv2=conv(conv_dim,conv_dim*2,4,batch_norm=False)
        self.conv3=conv(conv_dim*2,conv_dim*4,4,batch_norm=False)
        self.conv4=conv(conv_dim*4,conv_dim*8,4,batch_norm=False)

        self.fc=Linear(conv_dim*8*2*2,1)


    def forward(self,x):
        
        x=leaky_relu(self.conv1(x),0.2)
        x=leaky_relu(self.conv2(x),0.2)
        x=leaky_relu(self.conv3(x),0.2)
        x=leaky_relu(self.conv4(x),0.2)
        x = x.view(-1, self.conv_dim*8*2*2)
        out=self.fc(x)
        return out
    
class Generator(Module):
    def __init__(self,z_size,conv_dim):
        super(Generator,self).__init__()
        self.conv_dim=conv_dim
        self.fc = Linear(z_size,conv_dim*8*2*2)

        self.deconv1 = deconv(conv_dim*8,conv_dim*4,4)
        self.deconv2 = deconv(conv_dim*4,conv_dim*2,4)
        self.deconv3 = deconv(conv_dim*2,conv_dim,4)
        self.deconv4 = deconv(conv_dim,3,4,batch_norm=False)

    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2)
        x = relu(self.deconv1(x))
        x = relu(self.deconv2(x))
        x = relu(self.deconv3(x))
        x = self.deconv4(x)
        out = tanh(x)
        return out
    
def real_loss(D_out):
    batch_size = D_out.size(0)
        
    labels = ones(batch_size) * 0.9

    if use_cuda and is_available():
        labels = labels.cuda()
    criterion=BCEWithLogitsLoss()
    loss=criterion(D_out.squeeze(),labels)
    return loss

def fake_loss(D_out):
    batch_size = D_out.size(0)
        
    labels = ones(batch_size)

    if use_cuda and is_available():
            labels = labels.cuda()
    criterion=BCEWithLogitsLoss()
    loss=criterion(D_out.squeeze(),labels)
    return loss    
  
  
d_optimizer = Adam(D.parameters(),lr=0.0002,betas=(0.5,0.999))
g_optimizer = Adam(G.parameters(),lr=0.0002,betas=(0.5,0.999))
    