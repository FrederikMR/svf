#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 12:56:10 2023

@author: fmry
"""

#%% Sources

"""
Sources used:
https://github.com/ku2482/vae.pytorch/blob/master/models/simple_vae.py
https://github.com/sarthak268/Deep_Neural_Networks/blob/master/Autoencoder/Variational_Autoencoder/generative_vae.py
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
https://github.com/DeepLearningDTU/02456-deep-learning-with-PyTorch/blob/master/7_Unsupervised/7.2-EXE-variational-autoencoder.ipynb
"""

#%% Modules

from torch import (
    ones
    )

from torch.nn import (
    Module,
    BCEWithLogitsLoss,
    Sequential,
    Linear,
    BatchNorm1d,
    Identity,
    ELU
    )

from typing import List, Any

#%% Encoder

class Discriminator(Module):
    def __init__(self, 
                 input_dim,
                 ffh_layer:List[Any]
                 ):
        super(Discriminator, self).__init__()
        
        self.id, self.ffh_layer = input_dim, ffh_layer
                                                
        self.numh_layers = len(ffh_layer)
            
        self.encoder = self.encoder_layers()
                                                
    def encoder_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffh_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act())
        for i in range(1, self.numh_layers):
            out_feat, bias, batch, act = self.ffh_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act())
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, x):
        
        x_encoded = self.encoder(x)
        
        return x_encoded
    
#%% Decoder

class Generator(Module):
    def __init__(self,
                 input_dim:int,
                 ffg_layer:List[Any],
                 ):
        super(Generator, self).__init__()
    
        self.id, self.ffg_layer, self.numg_layers = input_dim, ffg_layer, len(ffg_layer)
        self.decoder = self.decoder_layers()
        
    def decoder_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffg_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act())
        for i in range(1, self.numg_layers):
            out_feat, bias, batch, act = self.ffg_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act())
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, z):
        
        return self.decoder(z)
        
#%% Feed Forward Variational Autoencoder

#The training script should be modified for the version below.
class FFGAN(Module):
    def __init__(self,
                 input_dim:int = 3,
                 ffh_layer:List[Any] = [[100, True, False, ELU]],
                 ffg_layer: List[int] = [[100, True, False, ELU], [3, True, False, Identity]],
                 criterion:Any = BCEWithLogitsLoss
                 ):
        super(FFGAN, self).__init__()
        
        self.encoder = Discriminator(input_dim, ffh_layer)
        self.decoder = Generator(ffh_layer[-1][0], ffg_layer)
        
        self.criterion = criterion
        
    def get_parameters(self):
        
        return self.discriminator.parameters(), self.generator.parameters()
    
    def real_loss(self, D_out):
        
        labels = ones(D_out.size(0)) * 0.9
        
        return self.criterion(D_out.squeeze(),labels)
    
    def fake_loss(self, D_out):
        
        labels = ones(D_out.size(0))
                        
        return self.criterion(D_out.squeeze(),labels)
    
    def forward(self, x, z):
        
        real_loss = self.real_loss(self.discriminator(x))
        fake_loss = self.fake_loss(self.generator(z))
        
        return real_loss, fake_loss, real_loss+fake_loss
        
    def g(self, z):
                
        return self.generator(z)



        
