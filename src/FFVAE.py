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

from torch.distributions import (
    Normal
    )

from torch import (
    Tensor,
    randn_like,
    sqrt,
    zeros_like,
    ones_like
    )

from torch.nn import (
    Module,
    Parameter,
    Sequential,
    Linear,
    BatchNorm1d,
    Identity,
    Sigmoid,
    ELU
    )

from typing import List, Any

#%% Encoder

class Encoder(Module):
    def __init__(self, 
                 input_dim,
                 ffh_layer:List[Any],
                 ffmu_layer:List[Any],
                 ffvar_layer:List[Any],
                 ):
        super(Encoder, self).__init__()
        
        self.id, self.ffh_layer, self.ffmu_layer, self.ffvar_layer = input_dim, \
            ffh_layer, ffmu_layer, ffvar_layer
                                                
        self.numh_layers, self.nummu_layers, self.numvar_layers = \
            len(ffh_layer), len(ffmu_layer), len(ffvar_layer)
            
        self.encoder, self.mu_net, self.var_net = self.encoder_layers(), self.mu_layers(), \
                                                    self.var_layers()
                                                
    def encoder_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffh_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.numh_layers):
            out_feat, bias, batch, act = self.ffh_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act)
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def mu_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffmu_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.nummu_layers):
            out_feat, bias, batch, act = self.ffmu_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act)
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def var_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffvar_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.numvar_layers):
            out_feat, bias, batch, act = self.ffvar_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act)
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def reparametrize(self, mu, std):
        
        eps = randn_like(std)
        z = mu + (eps * std)
        
        return z
    
    def forward(self, x):
        
        x_encoded = self.encoder(x)
        mu, std = self.mu_net(x_encoded), sqrt(self.var_net(x_encoded))
        z = self.reparametrize(mu, std)
        
        return z, mu, std
    
#%% Decoder

class Decoder(Module):
    def __init__(self,
                 input_dim:int,
                 ffg_layer:List[Any],
                 ):
        super(Decoder, self).__init__()
    
        self.id, self.ffg_layer, self.numg_layers = input_dim, ffg_layer, len(ffg_layer)
        self.decocer = self.decoder_layers()
        
    def decoder_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffg_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.numg_layers):
            out_feat, bias, batch, act = self.ffg_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act)
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, z):
        
        return self.decoder(z)
        
#%% Feed Forward Variational Autoencoder

#The training script should be modified for the version below.
class FFVAE(Module):
    def __init__(self,
                 input_dim:int = 3,
                 ffh_layer:List[Any] = [[100, True, False, ELU]],
                 ffmu_layer:List[Any] = [[2, True, False, Identity]],
                 ffvar_layer: List[int] = [[2, True, False, Sigmoid]],
                 ffg_layer: List[int] = [[100, True, False, ELU], [3, True, False, Identity]],
                 ):
        super(FFVAE, self).__init__()
        
        self.encoder = Encoder(input_dim, ffh_layer, ffmu_layer, ffvar_layer)
        self.decoder = Decoder(ffmu_layer[-1][0], ffg_layer)
        
        # for the gaussian likelihood
        self.exp_scale = Parameter(Tensor([1.0]))
        
    def gaussian_likelihood(self, x_hat, x):
        
        dist = Normal(x_hat, self.exp_scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        
        return log_pxz.sum(dim=1)

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p, q = Normal(zeros_like(mu), ones_like(std)), Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx, log_pz = q.log_prob(z), p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz).sum(-1)
        
        return kl
    
    def forward(self, x):
        
        z, mu, std = self.encoder(x)
        x_hat = self.decoder(z)
                
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kld, rec_loss = self.kl_divergence(z, mu, std).mean(), -self.gaussian_likelihood(x_hat, x).mean()
        
        # elbo
        elbo = kld + rec_loss
        
        return z, x_hat, mu, std, kld, rec_loss, elbo
    
    def h(self, x):
        
        return self.encoder(x)[1]
        
    def g(self, z):
                
        return self.decoder(z)



        
