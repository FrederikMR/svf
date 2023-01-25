# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:06:08 2021

@author: Frederik
"""

#%% Sources

"""
Sources:
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
http://adamlineberry.ai/vae-series/vae-code-experiments
https://gist.github.com/sbarratt/37356c46ad1350d4c30aefbd488a4faa
https://discuss.pytorch.org/t/cpu-ram-usage-increasing-for-every-epoch/24475/10
https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
"""

#%% Modules

from pandas import read_csv

from torch import (
    Tensor,
    transpose,
    load,
    save,
    )

from torch.cuda import (
    is_available
    )

from torch.utils.data import DataLoader

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

import datetime

#Own files
from FFVAE import FFVAE

#%% Parser for command line arguments

def parse_args():
    parser = ArgumentParser()
    # File-paths
    parser.add_argument('--data_name', default='hyperbolic_paraboloid', 
                        type=str)

    parser.add_argument('--latent_dim', default=2, type=int)

    #Hyper-parameters
    parser.add_argument('--epochs', default=10000, #100000
                        type=int)
    parser.add_argument('--batch_size', default=100,
                        type=int)
    parser.add_argument('--lr', default=0.0001,
                        type=float)
    parser.add_argument('--workers', default=0,
                        type=int)

    parser.add_argument('--save_hours', default=1,
                        type=int)
    #Continue training or not
    parser.add_argument('--con_training', default=0,
                        type=int)
    parser.add_argument('--load_epoch', default='100000.pt',
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
    
    if is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    
    #paths
    data_path = 'synthetic_data/'+args.data_name+'.csv'
    load_path = 'trained_models/'+args.data_name+'/'+args.data_name+'_epoch_'+args.load_epoch
    save_path = 'trained_models/'+args.data_name+'/'+args.data_name+'_epoch_'

    df = read_csv(data_path, index_col=0)
    DATA = Tensor(df.values).to(device) #DATA = torch.Tensor(df.values)
    DATA = transpose(DATA, 0, 1)

    if is_available():
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True, num_workers = args.workers)
    else:
        trainloader = DataLoader(dataset = DATA, batch_size= args.batch_size,
                                 shuffle = True, pin_memory=True, num_workers = args.workers)
        
    N = len(trainloader.dataset)
    
    input_dim = 3
    ffh_layer = [[100, True, False, ELU]]
    ffmu_layer = [[args.latent_dim, True, False, Identity]]
    ffvar_layer = [[args.latent_dim, True, False, Sigmoid]]
    ffg_layer = [[100, True, False, ELU], [3, True, False, Identity]]

    model = FFVAE(input_dim,
                  ffh_layer,
                  ffmu_layer,
                  ffvar_layer,
                  ffg_layer).to(device) #Model used

    optimizer = Adam(model.parameters(), lr=args.lr)

    if args.con_training:
        checkpoint = load(load_path, map_location=args.device)
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


    start_time = datetime.datetime.now()
    current_time = start_time
    time_diff = datetime.timedelta(hours=args.save_hours)
    
    model.train()
    for epoch in range(last_epoch, epochs):
        running_loss_elbo = 0.0
        running_loss_rec = 0.0
        running_loss_kld = 0.0
        for x in trainloader:
            #x = x.to(args.device) #If DATA is not saved to device
            _, x_hat, mu, var, kld, rec_loss, elbo = model(x)
            optimizer.zero_grad() #Based on performance tuning
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
            print(f"Saving Epoch {epoch+1}/{epochs} - loss: {train_epoch_loss:.4f}")
            checkpoint = save_path+str(epoch+1)+'.pt'
            save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ELBO': train_loss_elbo,
                'rec_loss': train_loss_rec,
                'KLD': train_loss_kld
                }, checkpoint)


    checkpoint = checkpoint = save_path+str(epoch+1)+'.pt'
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
