#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 21:01:57 2023

@author: frederik
"""

#%% Sources

#%% Modules

import numpy as np
import pandas as pd

#%% Hyper-parameters

np.random.seed(2712)

N_sim = 50000
data_path = 'synthetic_data/'

#%% Simulating Plane

mu = 0.0
std = 1.0

x1 = np.random.normal(mu, std, size=N_sim)
x2 = np.random.normal(mu, std, size=N_sim)
x3 = np.zeros(N_sim)

X = np.vstack((x1, x2, x3))
pd.DataFrame(X).to_csv(data_path+'plane.csv')

#%% Simulating Paraboloid

mu = 0.0
std = 1.0

x1 = np.random.normal(mu, std, size=N_sim)
x2 = np.random.normal(mu, std, size=N_sim)
x3 = x1**2+x2**2

X = np.vstack((x1, x2, x3))
pd.DataFrame(X).to_csv(data_path+'paraboloid.csv')

#%% Simulating Hyperbolic Paraboloid

mu = 0.0
std = 1.0

x1 = np.random.normal(mu, std, size=N_sim)
x2 = np.random.normal(mu, std, size=N_sim)
x3 = x1**2-x2**2

X = np.vstack((x1, x2, x3))
pd.DataFrame(X).to_csv(data_path+'hyperbolic_paraboloid.csv')

#%% Simulating circle

theta = np.random.uniform(0.0, 2*np.pi, size=N_sim)
x1 = np.cos(theta)
x2 = np.sin(theta)
x3 = np.zeros(N_sim)

X = np.vstack((x1, x2, x3))
pd.DataFrame(X).to_csv(data_path+'circle.csv')

#%% Simulating Sphere

r = 1.0
theta = np.random.uniform(0.0, 2*np.pi, size=N_sim)
phi = np.random.uniform(0.0, np.pi, size=N_sim)

x1 = r*np.cos(theta)*np.sin(phi)
x2 = r*np.sin(theta)*np.sin(phi)
x3 = r*np.sin(phi)

X = np.vstack((x1, x2, x3))
pd.DataFrame(X).to_csv(data_path+'sphere.csv')