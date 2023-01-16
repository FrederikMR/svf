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
    exp,
    prod,
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
                 channels_h:List[int],
                 kernel_size:List[int],
                 stride_h:List[int],
                 padding_h:List[int],
                 dilation_h:List[int],
                 groups_h:List[int],
                 padding_mode_h:List[str],
                 bias_h:List[bool],
                 batch_norm_h:List[bool],
                 ff_h:List[int],
                 ff_acth:List[Any],
                 latent_dim:int,
                 ):
        super(Encoder, self).__init__()
        
    def conv_layers(self):
        
        layers = []
        for i in range(1, self.num_conv_layers):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i-1],
                        stride = self.sh[i-1],
                        dilation = self.dh[i-1],
                        groups = self.gh[i-1],
                        bias = self.bh[i-1],
                        padding_mode = self.pmodh[i-1])

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.channels_h[i]))
            
        return Sequential(*layers)
        
#%% Decoder

class Decoder(Module):
    def __init__(self, 
                 ):
        super(Decoder, self).__init__()
        

#%% Deep Convolutional Variational-Autoencoder

class DCVAE(Module):
    def __init__(self,
                 input_dim = [32, 32],
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
                 ff_h:List[int] = [256],
                 ff_acth = [Identity],
                 latent_dim:int = 32,
                 ff_g:List[int] = [256],
                 ff_actg = [Identity],
                 act_g = [ELU, ELU, ELU, ELU, ELU],
                 channels_g = [64, 64, 32, 32, 32, 3],
                 kernel_size_g = [6, 4, 4, 4, 3],
                 stride_g = [2, 2, 2, 2, 1],
                 padding_g:List[int] = [0, 0, 0, 0],
                 dilation_g:List[int] = [1, 1, 1, 1],
                 batch_norm_g:List[bool] = [True, True, True, True],
                 ):
        super(DCVAE, self).__init__()
        
        self.num_conv = len(channels_h)
        self.num_linearh = len(ff_h)
        self.input_dim = input_dim
        self.acth = act_h
        self.ch = channels_h
        self.ksh = kernel_size_h
        self.sh = stride_h
        self.ph = padding_h
        self.dh = dilation_h
        self.gh = groups_h
        self.pmodh = padding_mode_h
        self.biash = bias_h
        self.bnormh = batch_norm_h
        self.linear_dim = prod(self.linear_dim, dtype=int)
        
        self.ff_h = ff_h
        self.ff_acth = ff_acth
        self.latent_dim = latent_dim
        self.ff_g = ff_g
        self.ff_actg = ff_actg
        
        self.actg = act_g
        self.cg = channels_g
        self.ksg = kernel_size_g
        self.sg = stride_g
        self.pg = padding_g
        self.dg = dilation_g
        self.bnormg = batch_norm_g
        
    def linear_dim(self):
        
        H_in, W_in = self.input_dim
        for i in range(1, self.num_conv_layers):
            pad, dil, ksize, stride = self.padding_h[i], self.dilation_h[i], \
                                        self.kernel_size_h[i], self.stride_h[i]
            
            pad_H, pad_W, dil_H, dil_W, ksize_H, ksize_W, stride_H, stride_W = \
                pad[0], pad[-1], dil[0], dil[-1], ksize[0], ksize[-1], stride[0], stride[-1]
                
            H_in = (H_in+2*pad_H-dil_H*(ksize_H-1)-1)/(stride_H)+1
            W_in = (W_in+2*pad_W-dil_W*(ksize_W-1)-1)/(stride_W)+1
            
        return H_in, W_in
        
    def conv_layers(self):
        
        layers = []
        for i in range(1, self.num_conv_layers):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i-1],
                        stride = self.sh[i-1],
                        dilation = self.dh[i-1],
                        groups = self.gh[i-1],
                        bias = self.bh[i-1],
                        padding_mode = self.pmodh[i-1])

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.channels_h[i]))
            
        return Sequential(*layers)
    
    def convt_layers(self):
        
        layers = []
        for i in range(1, self.num_conv_layers):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i-1],
                        stride = self.sh[i-1],
                        dilation = self.dh[i-1],
                        groups = self.gh[i-1],
                        bias = self.bh[i-1],
                        padding_mode = self.pmodh[i-1])

            layers.append(conv)
        
            if self.batch_norm_h[i-1]:
                layers.append(BatchNorm2d(self.channels_h[i]))
            
        return Sequential(*layers)
        
    def linear_hlayers(self):
        
        layer = []
        layer.append(self.linear_dim, self.ffh[0])
        layer.append(self.ffh_act[0]())
        for i in range(1, self.num_linearh-1):
            layer.append(Linear(self.ffh[i-1], self.ffh[i]))
            layer.append(self.ffh_act[i-1]())

        return Sequential(*layer)
    
    def linear_glayers(self):
        
        layer = []
        layer.append(self.latent_dim, self.ffg[0])
        layer.append(self.ffg_act[0]())
        for i in range(1, self.num_linearh-1):
            layer.append(Linear(self.ffg[i-1], self.ffg[i]))
            layer.append(self.ffg_act[i-1]())
            
        layer.append(Linear(self.fc_h[-1], self.latent_dim))
            
        return Sequential(*layer)
    
    def mu_layers(self):
        
        layer = []
        
        for i in range(1, self.num_fc_mu):
            layer.append(Linear(self.fc_mu[i-1], self.fc_mu[i]))
            layer.append(self.fc_mu_act[i-1]())
            
        return Sequential(*layer)
    
    def var_hlayers(self):
        
        layer = []
        
        for i in range(1, self.num_fc_var):
            layer.append(Linear(self.ff_var[i-1], self.ff_var[i]))
            layer.append(self.ff_var_act[i-1]())
            
        return Sequential(*layer)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        