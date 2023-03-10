# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 23:04:19 2021

@author: Frederik
"""

#%% Sources

"""
Sources:
https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

#%% Modules

from torch import (
    load,
    save,
    )

from torch.cuda import (
    is_available
    )

from torch.utils.data import DataLoader

from torch.utils.data import (
    Subset
    )

from torchvision.datasets import (
    SVHN
    )

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize
    )

from torch.optim import (
    Adam
    )

from torch.nn import (
    Identity,
    Sigmoid,
    ELU
    )

from argparse import (
    ArgumentParser
    )

from numpy.random import (
    choice
    )

import datetime

#Own files
from DC2DVAE import DC2DVAE

#%% Parser for command line arguments

def parse_args():
    parser = ArgumentParser()
    # File-paths
    parser.add_argument('--svhn_path', default="~/SynologyDrive/Cloudstation/Data/SVHN",
                        type=str)
    parser.add_argument('--save_model_path', default='trained_models/main/celeba', #'trained_models/surface_R2'
                        type=str)
    
    #Cropping image
    parser.add_argument('--img_size', default=64,
                        type=int)
    parser.add_argument('--num_img', default=0.8, #0.8
                        type=float)
    parser.add_argument('--train_type', default="train", #0.8
                        type=str)

    parser.add_argument('--workers', default=0, #2
                        type=int)
    parser.add_argument('--epochs', default=1000, #50000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0002,
                        type=float)
    
    parser.add_argument('--save_hours', default=1,
                        type=float)
    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--load_model_path', default='trained_models/main/celeba_epoch_5000.pt',
                        type=str)


    args = parser.parse_args()
    return args

#%% Main loop

def main():

    args = parse_args()
    train_loss_elbo = [] #Elbo loss
    train_loss_rec = [] #Reconstruction loss
    train_loss_kld = [] #KLD loss
    epochs = args.epochs
    time_diff = datetime.timedelta(hours=args.save_hours)
    start_time = datetime.datetime.now()
    current_time = start_time
    
    if is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    dataset = SVHN(root=args.svhn_path, split = args.train_type,
                           transform=Compose([
                               ToTensor(),
                           ]))
    
    dataset_subset = Subset(dataset, choice(len(dataset), int(args.num_img*len(dataset)), replace=False))

    if is_available():
        trainloader = DataLoader(dataset = dataset_subset, batch_size= args.batch_size,
                                 shuffle = True, num_workers = args.workers)
    else:
        trainloader = DataLoader(dataset = dataset_subset, batch_size= args.batch_size,
                                 shuffle = True, pin_memory=True, num_workers = args.workers)

    N = len(trainloader.dataset)

    model = DC2DVAE(input_dim = [3, 32, 32],
                    channels_h = [32, 32, 32, 64],
                    kernel_size_h = [[2,2], [2,2], [2,2], [2,2]],
                    channels_g = [64, 64, 32, 32, 3],
                    kernel_size_g = [[2,2], [2,2], [2,2], [2,2], [2,2]],
                    ffh_layer = [[256, True, True, ELU]],
                    ffmu_layer = [[32, True, False, Identity]],
                    ffvar_layer = [[32, True, False, Sigmoid]],
                    ffg_layer = [[256, True, True, Identity]],
                    stride_h = [[2,2],[2,2],[2,2],[2,2]],
                    padding_h = None,
                    dilation_h = None,
                    groups_h = None,
                    padding_mode_h = None,
                    bias_h = None,
                    batch_norm_h = None,
                    convh_act = [ELU, ELU, ELU, ELU],
                    stride_g = [[2,2], [2,2], [2,2], [2,2], [2,2]],
                    padding_g = None,
                    output_padding_g = None,
                    padding_mode_g = None,
                    groups_g = None,
                    bias_g = None,
                    dilation_g = None,
                    batch_norm_g = None,
                    convtg_act = [ELU, ELU, ELU, ELU, Identity]).to(device) #Model used

    optimizer = Adam(model.parameters(), lr=args.lr)

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = load(args.load_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_epoch = checkpoint['epoch']
        elbo = checkpoint['ELBO']
        rec_loss = checkpoint['rec_loss']
        kld_loss = checkpoint['KLD']

        train_loss_elbo = elbo
        train_loss_rec = rec_loss
        train_loss_kld = kld_loss
    else:
        last_epoch = 0

    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            dat = x[0].to(device)
            _, x_hat, mu, var, kld, rec_loss, elbo = model(dat)
            #optimizer.zero_grad(set_to_none=True) #Based on performance tuning
            optimizer.zero_grad()
            elbo.backward()
            optimizer.step()

            running_loss_elbo += elbo.item()
            running_loss_rec += rec_loss.item()
            running_loss_kld += kld.item()

            #del x, x_hat, mu, var, kld, rec_loss, elbo #In case you run out of memory

        train_epoch_loss = running_loss_elbo/N
        train_loss_elbo.append(train_epoch_loss)
        train_loss_rec.append(running_loss_rec/N)
        train_loss_kld.append(running_loss_kld/N)
        
        current_time = datetime.datetime.now()
        if current_time - start_time >= time_diff:
            print(f"Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
            checkpoint = args.save_model_path+'_epoch_'+str(epoch+1)+'.pt'
            save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)
            start_time = current_time


    checkpoint = args.save_model_path+'_epoch_'+str(epoch+1)+'.pt'
    save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)

    return

#%% Calling main

if __name__ == '__main__':
    main()
