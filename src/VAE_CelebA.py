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

#%% VAE for 3d surfaces using article network with variance and probabilities

#The training script should be modified for the version below.
class VAE_CELEBA(Module):
    def __init__(self,
                 latent_dim = 32
                 ):
        super(VAE_CELEBA, self).__init__()
                
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
        kld = self.kl_divergence(z, mu, std)
        rec_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        
        # elbo
        elbo = (kld - rec_loss).mean()
        
        return z, x_hat, mu, std, kld.mean(), -rec_loss.mean(), elbo
            
    def h(self, x):
        
        mu, _ = self.encoder(x)
        
        return mu
        
    def g(self, z):
        
        x_hat = self.decoder(z)
        
        return x_hat

#%% Simple test

"""
import torchvision.datasets as dset
import torchvision.transforms as transforms


dataroot = "../../Data/CelebA/celeba" #Directory for dataset
batch_size = 2 #Batch size duiring training
image_size = 64 #Image size
nc = 3 #Channels
vae = VAE_CELEBA() #Model

dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True)

for x in dataloader:
    test = vae(x[0])
    break
"""




        