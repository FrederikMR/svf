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
            = input_dim, channels, kernel_size, stride, padding, dilation, \
                groups, padding_mode, bias, batch_norm, ffh_layer, ffmu_layer, \
                    ffvar_layer
                
        self.num_conv, self.num_lin, self.num_mu, self.num_var = len(channels), \
            len(ffh_layer), len(ffmu_layer), len(ffvar_layer)
            
        self.convod, self.ld = self.linear_dim(), ffh_layer[-1][0]
        
        self.conv_encoder, self.lienar_encoder = self.conv_layers(), self.linear_layers()
    
    def linear_dim(self):
        
        _, H_in, W_in = self.id
        for i in range(1, self.num_conv):
            pad, dil, ksize, stride = self.ph[i], self.dh[i], self.ksh[i], self.sh[i]
            
            pad_H, pad_W, dil_H, dil_W, ksize_H, ksize_W, stride_H, stride_W = \
                pad[0], pad[-1], dil[0], dil[-1], ksize[0], ksize[-1], stride[0], stride[-1]
                
            H_in = (H_in+2*pad_H-dil_H*(ksize_H-1)-1)/(stride_H)+1
            W_in = (W_in+2*pad_W-dil_W*(ksize_W-1)-1)/(stride_W)+1
            
        return H_in, W_in
    
    def conv_layers(self):
        
        layers = []
        conv=Conv2d(in_channels = self.id[0],
                    out_channels = self.ch[0],
                    kernel_size = self.ksh[0],
                    stride = self.sh[0],
                    dilation = self.dh[0],
                    groups = self.gh[0],
                    bias = self.bh[0],
                    padding_mode = self.pmodh[0]
                    )
        layers.append(conv)
        if self.bnormh[0]:
            layers.append(BatchNorm2d(self.ch[0]))
            
        for i in range(1, self.num_conv):
            
            conv=Conv2d(in_channels = self.ch[i-1],
                        out_channels = self.ch[i],
                        kernel_size = self.ksh[i],
                        stride = self.sh[i],
                        dilation = self.dh[i],
                        groups = self.gh[i],
                        bias = self.bh[i],
                        padding_mode = self.pmodh[i]
                        )

            layers.append(conv)
        
            if self.batch_norm_h[i]:
                layers.append(BatchNorm2d(self.ch[i]))
            
        return Sequential(*layers)
    
    def linear_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffh_layer[0]
        layer.append(Linear(self.convod, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.num_lin):
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
        layer.append(Linear(self.ld, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.num_mu):
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
        layer.append(Linear(self.ld, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.num_var):
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
        
        x_encoded = self.linear_encoder(self.conv_layers(x).view(x.size[0], -1))
        mu, std = self.mu_net(x_encoded), sqrt(self.var_net(x_encoded))
        z = self.reparametrize(mu, std)
        
        return z, mu, std
        
#%% Decoder

class Decoder(Module):
    def __init__(self, 
                 input_dim:int,
                 ffg_layer:List[Any],
                 channels:List[int],
                 kernel_size:List[int],
                 stride:List[int],
                 padding:List[int],
                 output_padding:List[int],
                 padding_mode:List[str],
                 groups:List[int],
                 bias:List[bool],
                 dilation:List[int],
                 batch_norm:List[bool]
                 ):
        super(Decoder, self).__init__()
        
        self.id, self.cg, self.ksg, self.sg, self.pg, self.dg, self.gg, self.opg, \
        self.pmodg, self.bg, self.bnormg, self.ffg_layer \
            = input_dim, channels, kernel_size, stride, padding, dilation, \
                groups, output_padding, padding_mode, bias, batch_norm, ffg_layer
                
        self.num_tconv, self.num_lin, self.lin_dim = len(channels), len(ffg_layer), ffg_layer[-1][0]
        self.convt_encoder, self.linear_encoder = self.convt_layers(), self.linear_layers()
    
    def convt_layers(self):
        
        layers = []
        convt=ConvTranspose2d(in_channels = self.id,
                    out_channels = self.cg[0],
                    kernel_size = self.ksg[0],
                    stride = self.sg[0],
                    padding = self.pg[0],
                    output_padding = self.opg[0],
                    groups = self.gg[0],
                    bias = self.bg[0],
                    dilation = self.dg[0],
                    padding_mode = self.pmodg[0]
                    )
        layers.append(convt)
        if self.bnormg[0]:
            layers.append(BatchNorm2d(self.cg[0]))
        for i in range(1, self.num_tconv):
            
            convt=ConvTranspose2d(in_channels = self.cg[i-1],
                        out_channels = self.cg[i],
                        kernel_size = self.ksg[i],
                        stride = self.sg[i],
                        padding = self.pg[i],
                        output_padding = self.opg[i],
                        groups = self.gg[i],
                        bias = self.bg[i],
                        dilation = self.dg[i],
                        padding_mode = self.pmodg[i]
                        )

            layers.append(convt)
        
            if self.bnormg[i]:
                layers.append(BatchNorm2d(self.cg[i]))
            
        return Sequential(*layers)
    
    def linear_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffg_layer[0]
        layer.append(Linear(self.id, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act)
        for i in range(1, self.num_lin):
            out_feat, bias, batch, act = self.ffg_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act)
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, z):
        
        return self.decoder(self.linear_encoder(z).view(z.size(0), self.lin_dim, 1, 1))

#%% Deep Convolutional Variational-Autoencoder

class DC2DVAE(Module):
    def __init__(self,
                 input_dim:List[int],
                 channels_h:List[int],
                 kernel_size_h:List[int],
                 channels_g:List[int],
                 kernel_size_g:List[int],
                 ffh_layer:List[Any],
                 ffmu_layer:List[Any],
                 ffvar_layer:List[Any],
                 ffg_layer:List[Any],
                 stride_h:List[int] = None,
                 padding_h:List[int] = None,
                 dilation_h:List[int] = None,
                 groups_h:List[int] = None,
                 padding_mode_h:List[str] = None,
                 bias_h:List[bool] = None,
                 batch_norm_h:List[bool] = None,
                 stride_g:List[int] = None,
                 padding_g:List[int] = None,
                 output_padding_g:List[int] = None,
                 padding_mode_g:List[str] = None,
                 groups_g:List[int] = None,
                 bias_g:List[bool] = None,
                 dilation_g:List[int] = None,
                 batch_norm_g:List[bool] = None
                 ):
        super(DC2DVAE, self).__init__()
        
        num_convh = len(channels_h)
        num_convg = len(channels_g)
        
        if stride_h is None:
            stride_h = [[1,1]]*num_convh
        if padding_h is None:
            padding_h = [[0,0]]*num_convh
        if dilation_h is None:
            dilation_h = [[1,1]]*num_convh
        if groups_h is None:
            groups_h = [[1,1]]*num_convh
        if padding_mode_h is None:
            padding_mode_h = ['zeros']*num_convh
        if bias_h is None:
            bias_h = [True]*num_convh
        if batch_norm_h is None:
            batch_norm_h = [True]*num_convh
            
        if stride_g is None:
            stride_g = [[1,1]]*num_convg
        if padding_g is None:
            padding_g = [[0,0]]*num_convg
        if output_padding_g is None:
            output_padding_g = [[0,0]]*num_convg
        if groups_g is None:
            groups_g = [[1,1,]]*num_convg
        if bias_g is None:
            bias_g = [True]*num_convg
        if dilation_g is None:
            dilation_g = [[1,1]]*num_convg
        if padding_mode_g is None:
            padding_mode_g = ['zeros']*num_convg        
        
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
        
        self.decoder = Decoder(ffmu_layer[-1][0],
                                ffg_layer,
                                channels_g,
                                kernel_size_g,
                                stride_g,
                                padding_g,
                                output_padding_g,
                                padding_mode_g,
                                groups_g,
                                bias_g,
                                dilation_g,
                                batch_norm_g
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
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        