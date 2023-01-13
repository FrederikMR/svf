# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:18:47 2021

@author: Frederik
"""

#%% Sources

"""
Sources used:
https://github.com/bhpfelix/Variational-Autoencoder-PyTorch/blob/master/src/vanila_vae.py
https://arxiv.org/abs/1312.6114
https://github.com/sksq96/pytorch-vae/blob/master/vae.py
"""


#%% Modules

from torch.nn import (
    Module, 
    Sequential,
    Conv2d, 
    BatchNorm2d, 
    Linear, 
    BatchNorm1d, 
    ConvTranspose2d,
    ELU,
    Sigmoid,
    Parameter,
    )

from torch import (
    Tensor,
    randn_like,
    exp,
    ones_like,
    zeros_like
    )

from torch.distributions import (
    Normal
    )

from typing import List, Any

#%% Deep Convolutional Variational-Autoencoder

class DCVAE(Module):
    def __init__(self,
                 act_h = [ELU, ELU, ELU, ELU, ELU],
                 channels_h:List[int] = [3, 32, 32, 64],
                 kernel_size_h:List[int] = [4, 4, 4, 4],
                 stride_h:List[int] = [2, 2, 2, 2],
                 padding_h:List[int] = [0, 0, 0, 0],
                 dilation_h:List[int] = [1, 1, 1, 1],
                 groups_h:List[int] = [1, 1, 1, 1],
                 padding_mode_h = ['zeros', 'zeros', 'zeros', 'zeros'],
                 bias_h:List[bool] = [False, False, False, False],
                 batch_norm_h:List[bool] = [True, True, True, True],
                 latent_dim:int = 32,
                 act_g = [ELU, ELU, ELU, ELU, ELU],
                 channels_g = [64, 64, 32, 32, 32, 3],
                 kernel_size_g = [6, 4, 4, 4, 3],
                 stride_g = [2, 2, 2, 2, 1],
                 padding_g:List[int] = [0, 0, 0, 0],
                 dilation_g:List[int] = [1, 1, 1, 1],
                 batch_norm_g:List[bool] = [True, True, True, True],
                 ):
        super(DCVAE, self).__init__()
        
        self.latent_dim = 32
        
        #Encoder
        self.act_h = act_h
        self.channels_h = channels_h
        self.kernel_size_h = kernel_size_h
        self.stride_h = stride_h
        self.bias_h = bias_h
        self.batch_norm_h = batch_norm_h
        self.num_layers_h = len(act_h)
        
        #Decoder
        self.act_g = act_g
        self.channels_g = channels_g
        self.kernel_size_g = kernel_size_g
        self.stride_g = stride_g
        self.batch_norm_g = batch_norm_g
        self.num_layers_h = len(act_g)
                
        #Encoder
        self.h_con1 = Conv2d(in_channels = 3, out_channels = 32, kernel_size = 4, 
                             stride = 2, bias = False) #31x31z32
        self.h_batch1 = BatchNorm2d(32)
        
        self.h_con2 = Conv2d(in_channels = 32, out_channels = 32, kernel_size = 4, 
                             stride = 2, bias = False) #14x14x32
        self.h_batch2 = BatchNorm2d(32)
        
        self.h_con3 = Conv2d(in_channels = 32, out_channels = 64, kernel_size = 4, 
                             stride = 2, bias = False) #6x6x64
        self.h_batch3 = BatchNorm2d(64)
        
        self.h_con4 = Conv2d(in_channels = 64, out_channels = 64, kernel_size = 4, 
                             stride = 2, bias = False) #2x2x64
        self.h_batch4 = BatchNorm2d(64)
        
        self.h_fc = Linear(256, 256)
        self.h_batch5 = BatchNorm1d(256)
        
        #Mean and std
        self.h_mean = Linear(256, latent_dim)
        self.h_std = Linear(256, latent_dim)
        
        #Decoder
        self.g_fc = Linear(latent_dim, 256)
        self.g_batch1 = BatchNorm1d(256)
        
        self.g_tcon1 = ConvTranspose2d(in_channels = 256, out_channels = 64, 
                                       kernel_size = 6, stride = 2)
        self.g_batch2 = BatchNorm2d(64)
        
        self.g_tcon2 = ConvTranspose2d(in_channels = 64, out_channels = 64, 
                                       kernel_size = 4, stride = 2)
        self.g_batch3 = BatchNorm2d(64)
        
        self.g_tcon3 = ConvTranspose2d(in_channels = 64, out_channels = 32, 
                                       kernel_size = 4, stride = 2)
        self.g_batch4 = BatchNorm2d(32)
        
        self.g_tcon4 = ConvTranspose2d(in_channels = 32, out_channels = 32, 
                                       kernel_size = 4, stride = 2)
        self.g_batch5 = BatchNorm2d(32)
        
        self.g_tcon5 = ConvTranspose2d(in_channels = 32, out_channels = 3, 
                                       kernel_size = 3, stride = 1)

        self.ELU = ELU()
        self.Sigmoid = Sigmoid()
        
        # for the gaussian likelihood
        self.log_scale = Parameter(Tensor([0.0]))
        
    def linear_dim(self):
        
        C
        for i in range(1, self.num_layers_h):
            
        return
        
    def h_layers(self):
        
        layers = []
        for i in range(1, self.num_layers_h):
            
            conv=Conv2d(in_channels = self.channels_h[i-1],
                        out_channels = self.channels_h[i],
                        kernel_size = self.ks_h[i-1],
                        stride = self.stride_h[i-1],
                        dilation = self.dilation_h[i-1],
                        groups = self.groups_h[i-1],
                        bias = self.bias_h[i-1],
                        padding_mode = self.padding_mode[i-1])

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.channels_h[i]))
            
        return Sequential(*layers)
                
    def encoder(self, x):
                
        x1 = self.ELU(self.h_batch1(self.h_con1(x)))
        x2 = self.ELU(self.h_batch2(self.h_con2(x1)))
        x3 = self.ELU(self.h_batch3(self.h_con3(x2)))
        x4 = self.ELU(self.h_batch4(self.h_con4(x3)))
        
        x4 = x4.view(x4.size(0), -1)
        x5 = self.ELU(self.h_batch5(self.h_fc(x4)))
        
        mu = self.h_mean(x5)
        std = self.Sigmoid(self.h_std(x5))
        
        return mu, std
        
    def rep_par(self, mu, std):
        
        eps = randn_like(std)
        z = mu + (std*eps)
        
        return z
        
    def decoder(self, z):
                
        x1 = self.ELU(self.g_batch1(self.g_fc(z)))
        x1 = x1.view(-1, 256, 1, 1)
        
        x2 = self.ELU(self.g_batch2(self.g_tcon1(x1)))
        x3 = self.ELU(self.g_batch3(self.g_tcon2(x2)))
        x4 = self.ELU(self.g_batch4(self.g_tcon3(x3)))
        x5 = self.ELU(self.g_batch5(self.g_tcon4(x4)))
        
        x_hat = self.g_tcon5(x5)
        
        return x_hat
    
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = exp(logscale)
        dist = Normal(x_hat, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        
        return log_pxz.sum(dim=(1,2,3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = Normal(zeros_like(mu), ones_like(std))
        q = Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        
        return kl
        
    def forward(self, x):
        
        mu, std = self.encoder(x)
        
        z = self.rep_par(mu, std)
        
        x_hat = self.decoder(z)
        
        # compute the ELBO with and without the beta parameter: 
        # `L^\beta = E_q [ log p(x|z) - \beta * D_KL(q(z|x) | p(z))`
        # where `D_KL(q(z|x) | p(z)) = log q(z|x) - log p(z)`
        kld = self.kl_divergence(z, mu, std).mean()
        rec_loss = -self.gaussian_likelihood(x_hat, self.log_scale, x).mean()
        
        # elbo
        elbo = kld + rec_loss
        
        return z, x_hat, mu, std, kld, rec_loss, elbo
            
    def h(self, x):
        
        mu, _ = self.encoder(x)
        
        return mu
        
    def g(self, z):
        
        x_hat = self.decoder(z)
        
        return x_hat




        