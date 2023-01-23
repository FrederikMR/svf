# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 01:08:27 2021

@author: Frederik
"""

import numpy as np
import pandas as pd 
import torch

def x1_fun(N, mu = 0, std = 1):
    
    x1 = np.random.normal(mu, std, N)
    
    return x1

def x2_fun(N, mu = 0, std = 1):
    
    x2 = np.random.normal(mu, std, N)
    
    return x2
    
def x3_fun(x1, x2):
    
    return x1, x2, x1**2-x2**2

class sim_3d_fun(object):
    def __init__(self,
                 x1_fun = x1_fun,
                 x2_fun = x2_fun, 
                 x3_fun = x3_fun,
                 N_sim = 50000,
                 name_path = 'para_data.csv'):
        
        self.x1_fun = x1_fun
        self.x2_fun = x2_fun
        self.x3_fun = x3_fun
        self.N_sim = N_sim
        self.name_path = name_path
        
    def sim_3d(self):
    
        #np.random.seed(self.seed)
        x1 = self.x1_fun(self.N_sim)
        x2 = self.x2_fun(self.N_sim)
        
        x1, x2, x3 = self.x3_fun(x1, x2)
        
        df = np.vstack((x1, x2, x3))
        
        pd.DataFrame(df).to_csv(self.name_path)
        
        return
    
    def read_data(self):
        
        df = pd.read_csv(self.name_path, index_col=0)

        dat = torch.Tensor(df.values)
        
        dat = torch.transpose(dat, 0, 1)
        
        return dat
        
        
        
        

        