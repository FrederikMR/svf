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
                 ffd_layer:List[Any]):
        super(Discriminator,self).__init__()
        
        self.id, self.cd, self.ksd, self.sd, self.pd, self.dd, self.gd, self.pmodd, \
        self.bd, self.bnormd, self.ffd_layer \
            = input_dim, channels, kernel_size, stride, padding, dilation, \
                groups, padding_mode, bias, batch_norm, ffd_layer
                
        self.num_conv, self.convod = len(channels), self.linear_dim()
        
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

    def forward(self,x):
        
        x_encoded = self.linear_encoder(self.conv_layers(x).view(x.size[0], -1))
        
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
        
        return self.decoder(self.linear_encoder(z).view(z.size(0), self.lin_dim, 1, 1))
        
    def forward(self,x):
        x = self.fc(x)
        x = x.view(-1, self.conv_dim*8, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = self.deconv4(x)
        out = F.tanh(x)
        return out