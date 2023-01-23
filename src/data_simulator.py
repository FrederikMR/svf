#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:59:04 2023

@author: frederik
"""

#%% Sources

#%% Modules

#%% Functions

def sim_data(x, x_fun, f_fun):

    #np.random.seed(self.seed)
    x1 = self.x1_fun(self.N_sim)
    x2 = self.x2_fun(self.N_sim)
    
    x1, x2, x3 = self.x3_fun(x1, x2)
    
    df = np.vstack((x1, x2, x3))
    
    pd.DataFrame(df).to_csv(self.name_path)
    
    return