#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 10:35:19 2023

@author: fmry
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
    Identity,
    Parameter,
    )

from torch import (
    Tensor,
    randn_like,
    sqrt,
    ones_like,
    zeros_like
    )

from torch.distributions import (
    Normal
    )

from typing import List, Any

#%% Encoder

class Encoder(Module):
    def __init__(self,
                 input_dim:List[int],
                 channels:List[int],
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 padding_mode:List[str],
                 dilation:List[int],
                 groups:List[int],
                 bias:List[bool],
                 batch_norm:List[bool],
                 ffh_layer:List[Any],
                 ffmu_layer: List[Any],
                 ffvar_layer: List[Any],
                 ffvar_act: List[Any]
                 ):
        super(Encoder, self).__init__()
        
        self.id, self.ch, self.ksh, self.sh, self.ph, self.dh, self.gh, self.pmodh, \
        self.bh, self.bnormh, self.ffh_layer, self.ffmu_layer, self.ffvar_layer \
            = input_dim, channels_h, kernel_size_h, stride_h, padding_h, dilation_h, \
                groups_h, padding_mode_h, bias_h, batch_norm_h, ff_layersh, ff_acth, \
                ffmu_layer, ffvar_layer, ffmu_act, ffvar_act
                
        self.num_conv, self.num_lin, self.num_mu, self.num_var = len(channels_h), \
            len(ff_layersh), len(ffmu_layer), len(ffvar_layer)
            
        self.convod = self.linear_dim()
        
        self.conv_encoder, self.lienar_encoder = self.conv_layers(), self.linear_layers()
    
    def linear_dim(self):
        
        H_in, W_in = self.id
        for i in range(1, self.num_conv):
            pad, dil, ksize, stride = self.ph[i], self.dh[i], self.ksh[i], self.sh[i]
            
            pad_H, pad_W, dil_H, dil_W, ksize_H, ksize_W, stride_H, stride_W = \
                pad[0], pad[-1], dil[0], dil[-1], ksize[0], ksize[-1], stride[0], stride[-1]
                
            H_in = (H_in+2*pad_H-dil_H*(ksize_H-1)-1)/(stride_H)+1
            W_in = (W_in+2*pad_W-dil_W*(ksize_W-1)-1)/(stride_W)+1
            
        return H_in, W_in
    
    def conv_layers(self):
        
        layers = []
        for i in range(1, self.num_conv):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i-1],
                        stride = self.sh[i-1],
                        dilation = self.dh[i-1],
                        groups = self.gh[i-1],
                        bias = self.bh[i-1],
                        padding_mode = self.pmodh[i-1]
                        )

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.ch[i]))
            
        return Sequential(*layers)
    
    def linear_layers(self):
        
        layers = []
        layers.append(Linear(self.convod, self.ffh_layers[0]))
        layers.append(self.ffh_act[0]())
        for i in range(1, self.num_lin):
            layers.append(Linear(self.ffh[i-1], self.ffh_layers[i]))
            layers.append(self.ffh_act[i-1]())

        return Sequential(*layers)
    
    def mu_layers(self):
        
        layers = []
        for i in range(1, self.nummu_layers):
            layers.append(Linear(self.ffmu_layer[i-1], self.ffmu_layer[i]))
            layers.append(self.ffmu_act[i-1]())
            
        return Sequential(*layers)
    
    def var_layers(self):
        
        layers = []
        for i in range(1, self.numvar_layers):
            layers.append(Linear(self.fc_var[i-1], self.fc_var[i]))
            layers.append(self.fc_var_act[i-1]())
            
        return Sequential(*layers)
    
    def reparametrize(self, mu, std):
        
        eps = randn_like(std)
        z = mu + (eps * std)
        
        return z
    
    def forward(self, x):
        
        x_encoded = self.linear_encoder(self.conv_layers(x).view(x.size[0], -1))
        mu, std = self.mu_net(x_encoded), sqrt(self.var_net(x_encoded))
        z = self.reparametrize(mu, std)
        
        return z, mu, std
        
#%% Decoder

class Decoder(Module):
    def __init__(self, 
                 output_dim:int,
                 ffg_layer:List[Any],
                 channels_g:List[int],
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 output_padding:List[int],
                 groups:List[int],
                 bias:List[bool],
                 dilation:List[int],
                 batch_norm:List[bool]
                 ):
        super(Decoder, self).__init__()
        
        self.od, self.cg, self.ksg, self.sg, self.pg, self.dg, self.gg, self.opg, \
        self.bg, self.bnormg, self.ffg_layers, self.ffg_act \
            = output_dim, channels_g, kernel_size, stride, padding, dilation, \
                groups, output_padding, bias, batch_norm, ff_layersg, ff_actg
                
        self.num_tconv, self.num_lin, self.lin_dim = len(channels_g), len(ff_layersg), ff_layersg[-1]
        self.convt_encoder, self.lienar_encoder = self.convt_layers(), self.linear_layers()
    
    def convt_layers(self):
        
        layers = []
        for i in range(1, self.num_conv):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i-1],
                        stride = self.sh[i-1],
                        dilation = self.dh[i-1],
                        groups = self.gh[i-1],
                        bias = self.bh[i-1],
                        padding_mode = self.pmodh[i-1]
                        )

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.ch[i]))
            
        return Sequential(*layers)
    
    def linear_layers(self):
        
        layers = []
        layers.append(Linear(self.convod, self.ffh_layers[0]))
        layers.append(self.ffh_act[0]())
        for i in range(1, self.num_lin):
            layers.append(Linear(self.ffh[i-1], self.ffh_layers[i]))
            layers.append(self.ffh_act[i-1]())

        return Sequential(*layers)
    
    def forward(self, z):
        
        return self.decoder(self.linear_encoder(z).view(z.size(0), self.lin_dim, 1, 1))

#%% Deep Convolutional Variational-Autoencoder

class DC2DVAE(Module):
    def __init__(self,
                 input_dim:List[int],
                 channels_h:List[int],
                 kernel_size_h:List[int],
                 stride_h:List[int],
                 padding_h:List[int],
                 dilation_h:List[int],
                 groups_h:List[int],
                 padding_mode_h:List[str],
                 bias_h:List[bool],
                 batch_norm_h:List[bool],
                 ffh_layer:List[Any],
                 ffmu_layer: List[Any],
                 ffvar_layer: List[Any],
                 ffg_layer:List[Any],
                 channels_g:List[int],
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 output_padding:List[int],
                 groups:List[int],
                 bias:List[bool],
                 dilation:List[int],
                 batch_norm:List[bool]
                 ):
        super(DC2DVAE, self).__init__()
        
        self.encoder = Encoder(input_dim,
                                channels_h,
                                kernel_size_h,
                                stride_h,
                                padding_h,
                                dilation_h,
                                groups_h,
                                padding_mode_h,
                                bias_h,
                                batch_norm_h,
                                ffh_layer,
                                ffmu_layer,
                                ffvar_layer
                                )
        
        self.decoder = Decoder(output_dim,
                                ffg_layer,
                                channels_g,
                                kernel_size,
                                stride,
                                padding,
                                output_padding,
                                groups,
                                bias,
                                dilation,
                                batch_norm
                                )
        
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
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        