#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:06:11 2023

@author: frederikrygaard
"""

#%% Sources

#https://towardsdatascience.com/celebrity-face-generation-with-deep-convolutional-gans-40b96147a1c9

#%% Modules

from torch.nn import (
    Module, 
    Sequential,
    BCEWithLogitsLoss,
    Conv2d, 
    BatchNorm2d, 
    Linear, 
    BatchNorm1d, 
    ConvTranspose2d,
    Identity
    )

from torch import (
    Tensor,
    prod,
    ones,
    )

from typing import List, Any

#%% Discriminator

class Discriminator(Module):
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
                 conv_act:List[Any],
                 ffh_layer:List[Any]
                 ):
        super(Discriminator, self).__init__()
        
        self.id, self.ch, self.ksh, self.sh, self.ph, self.dh, self.gh, self.pmodh, \
        self.bh, self.bnormh, self.conv_act, self.ffh_layer \
            = input_dim, channels, kernel_size, stride, padding, dilation, \
                groups, padding_mode, bias, batch_norm, conv_act, ffh_layer
                
        self.num_conv, self.num_lin = len(channels), len(ffh_layer)
            
        self.convod, self.ld = int(channels[-1]*prod(self.linear_dim())), ffh_layer[-1][0]
        
        self.conv_encoder, self.linear_encoder, self.mu_net, self.var_net = \
            self.conv_layers(), self.linear_layers(), self.mu_layers(), self.var_layers()
    
    def linear_dim(self):
        
        _, H_in, W_in = self.id
        for i in range(0, self.num_conv):
            pad, dil, ksize, stride = self.ph[i], self.dh[i], self.ksh[i], self.sh[i]
            
            pad_H, pad_W, dil_H, dil_W, ksize_H, ksize_W, stride_H, stride_W = \
                pad[0], pad[-1], dil[0], dil[-1], ksize[0], ksize[-1], stride[0], stride[-1]
                
            H_in = int((H_in+2*pad_H-dil_H*(ksize_H-1)-1)/(stride_H)+1)
            W_in = int((W_in+2*pad_W-dil_W*(ksize_W-1)-1)/(stride_W)+1)
            
        return Tensor([H_in, W_in])
    
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
        layers.append(self.conv_act[0]())
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
            
            layers.append(self.conv_act[i]())
        
            if self.bnormh[i]:
                layers.append(BatchNorm2d(self.ch[i]))
            
        return Sequential(*layers)
    
    def linear_layers(self):
        
        layer = []
        in_feat, bias, batch, act = self.ffh_layer[0]
        layer.append(Linear(self.convod, in_feat, bias))
        if batch:
            layer.append(BatchNorm1d(in_feat))
        layer.append(act())
        for i in range(1, self.num_lin):
            out_feat, bias, batch, act = self.ffh_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act())
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, x):
        
        x_encoded = self.linear_encoder(self.conv_encoder(x).view(x.size(0), -1))
        
        return x_encoded
    
#%% Generator

class Generator(Module):
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
                 batch_norm:List[bool],
                 convt_act:List[Any]
                 ):
        super(Generator, self).__init__()
        
        self.id, self.cg, self.ksg, self.sg, self.pg, self.dg, self.gg, self.opg, \
        self.pmodg, self.bg, self.bnormg, self.convt_act, self.ffg_layer \
            = input_dim, channels, kernel_size, stride, padding, dilation, \
                groups, output_padding, padding_mode, bias, batch_norm, convt_act, ffg_layer
                
        self.num_tconv, self.num_lin, self.lin_dim = len(channels), len(ffg_layer), ffg_layer[-1][0]
        self.convt_encoder, self.linear_encoder = self.convt_layers(), self.linear_layers()
    
    def convt_layers(self):
        
        layers = []
        convt=ConvTranspose2d(in_channels = self.lin_dim,
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
        layers.append(self.convt_act[0]())
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
            layers.append(self.convt_act[i]())
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
        layer.append(act())
        for i in range(1, self.num_lin):
            out_feat, bias, batch, act = self.ffg_layer[i]
            layer.append(Linear(in_feat, out_feat, bias))
            if batch:
                layer.append(BatchNorm1d(out_feat))
            layer.append(act())
            in_feat = out_feat
            
        return Sequential(*layer)
    
    def forward(self, z):
        
        return self.convt_encoder(self.linear_encoder(z).view(z.size(0), self.lin_dim, 1, 1))
    
#%% Deep Convolutional Generative Adversarial Network

class DC2DGAN(Module):
    def __init__(self,
                 input_dim:List[int],
                 channels_h:List[int],
                 kernel_size_h:List[int],
                 channels_g:List[int],
                 kernel_size_g:List[int],
                 ffh_layer:List[Any],
                 ffg_layer:List[Any],
                 stride_h:List[int] = None,
                 padding_h:List[int] = None,
                 dilation_h:List[int] = None,
                 groups_h:List[int] = None,
                 padding_mode_h:List[str] = None,
                 bias_h:List[bool] = None,
                 batch_norm_h:List[bool] = None,
                 convh_act:List[Any] = None,
                 stride_g:List[int] = None,
                 padding_g:List[int] = None,
                 output_padding_g:List[int] = None,
                 padding_mode_g:List[str] = None,
                 groups_g:List[int] = None,
                 bias_g:List[bool] = None,
                 dilation_g:List[int] = None,
                 batch_norm_g:List[bool] = None,
                 convtg_act:List[Any] = None,
                 criterion:Any = BCEWithLogitsLoss
                 ):
        super(DC2DGAN, self).__init__()
        
        num_convh = len(channels_h)
        num_convg = len(channels_g)
        
        if stride_h is None:
            stride_h = [[1,1]]*num_convh
        if padding_h is None:
            padding_h = [[0,0]]*num_convh
        if dilation_h is None:
            dilation_h = [[1,1]]*num_convh
        if groups_h is None:
            groups_h = [1]*num_convh
        if padding_mode_h is None:
            padding_mode_h = ['zeros']*num_convh
        if bias_h is None:
            bias_h = [True]*num_convh
        if batch_norm_h is None:
            batch_norm_h = [True]*num_convh
        if convh_act is None:
            convh_act = [Identity]*num_convh
        
        if stride_g is None:
            stride_g = [[1,1]]*num_convg
        if padding_g is None:
            padding_g = [[0,0]]*num_convg
        if output_padding_g is None:
            output_padding_g = [[0,0]]*num_convg
        if groups_g is None:
            groups_g = [1]*num_convg
        if bias_g is None:
            bias_g = [True]*num_convg
        if dilation_g is None:
            dilation_g = [[1,1]]*num_convg
        if padding_mode_g is None:
            padding_mode_g = ['zeros']*num_convg    
        if batch_norm_g is None:
            batch_norm_g = [True]*num_convg
        if convtg_act is None:
            convtg_act = [Identity]*num_convg
        
        self.discriminator = Discriminator(input_dim,
                                channels_h,
                                kernel_size_h,
                                stride_h,
                                padding_h,
                                padding_mode_h,
                                dilation_h,
                                groups_h,
                                bias_h,
                                batch_norm_h,
                                convh_act,
                                ffh_layer
                                )
        
        self.generator = Generator(ffh_layer[-1][0],
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
                                batch_norm_g,
                                convtg_act
                                )
        
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
    