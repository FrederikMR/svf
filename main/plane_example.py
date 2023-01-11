#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  8 19:19:54 2023

@author: frederik
"""

#%% Sources

#%% Modules

import jax.numpy as jnp
from jax import vmap
import matplotlib.pyplot as plt

#Own modules
import svrm
import sp
import rm

#%% Functions

def param_fun(x):
    
    return jnp.array([x[0], x[1], 0.0])

def mu_fun(x):
    
    return 1.0

def sigma_fun(x):
    
    return jnp.exp(-jnp.dot(x,x))

#%% Hyper-parameters

N_sim = 10
n_points = 100

x1_grid = jnp.linspace(-2,2,n_points)
x2_grid = jnp.linspace(-2,2,n_points)
X1, X2 = jnp.meshgrid(x1_grid, x2_grid)
X3 = jnp.zeros_like(X1)
X = jnp.transpose(jnp.concatenate((X1.reshape(1,n_points, n_points), 
                                   X2.reshape(1,n_points, n_points))), axes=(1,2,0))

eta = sp.sim_normal(0,1,N_sim)

x0 = jnp.array([-1.0, 1.0])
xT = jnp.array([1.0, -1.0])
v0 = jnp.array([0.5, -0.5])

#%% Defining Geometry

RM_SG = svrm.gp_svfrm(mu_fun, sigma_fun, 2, param_fun = param_fun)
RM_G = rm.rm_geometry(param_fun = param_fun)

#%% Curvature

sc_sg = vmap(lambda e: vmap(lambda x: vmap(lambda y: RM_SG.SC(y, jnp.array([1.0,0.0]), jnp.array([0.0, 1.0]), e))(x))(X))(eta)
sc_g = vmap(lambda x: vmap(lambda y: RM_G.SC(y, jnp.array([1.0,0.0]), jnp.array([0.0, 1.0])))(x))(X)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X[:,:,0], X[:,:,1], sc_g, color='red', alpha=0.8)
for i in range(N_sim):
    ax.plot_surface(X[:,:,0], X[:,:,1], sc_sg[i], color='cyan', alpha=0.2)

plt.tight_layout()
plt.show()

#%% IVP plot

gammaG, _ = RM_G.geo_ivp(x0,v0)
gammaG_manifold = vmap(lambda x: param_fun(x))(gammaG)

gammaSG = vmap(lambda x: RM_SG.geo_ivp(x0, v0, x)[0])(eta)
gammaSG_manifold = vmap(lambda y: vmap(lambda x: param_fun(x))(y))(gammaSG)

#3d plot
plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")

ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

ax.plot(gammaG_manifold[:,0], gammaG_manifold[:,1], gammaG_manifold[:,2], 
        label='True Geodesics', color='red')

for i in range(N_sim-1):
    ax.plot(gammaSG_manifold[i][:, 0], gammaSG_manifold[i][:, 1], 
            gammaSG_manifold[i][:, 2],  color='cyan')
ax.plot(gammaSG_manifold[N_sim-1][:, 0], gammaSG_manifold[N_sim-1][:, 1], 
        gammaSG_manifold[N_sim-1][:, 2], 
        label='Stochastic Geodesics', color='cyan')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()

#2d plot
plt.figure(figsize=(8,6))

plt.plot(gammaG[:,0], gammaG[:,1], '-*', label='True Geodesic', color='red')

for i in range(N_sim-1):
    plt.plot(gammaSG[i][:,0], gammaSG[i][:,1], color='cyan')
plt.plot(gammaSG[N_sim-1][:,0], gammaSG[N_sim-1][:,1], label='Stochastic Geodesics', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()

plt.show()

#%% BVP plot

gammaG, _ = RM_G.geo_bvp(x0,xT)
gammaG_manifold = vmap(lambda x: param_fun(x))(gammaG)

gammaSG = []
for i in range(N_sim):
    gammaSG.append(RM_SG.geo_bvp(x0, xT, eta[i])[0])

gammaSG = jnp.stack(gammaSG)
gammaSG_manifold = vmap(lambda y: vmap(lambda x: param_fun(x))(y))(gammaSG)

#3d plot
plt.figure(figsize=(8,6))
ax = plt.axes(projection="3d")

ax.plot_surface(
X1, X2, X3,  rstride=1, cstride=1, color='c', alpha=0.2, linewidth=0)

ax.plot(gammaG_manifold[:,0], gammaG_manifold[:,1], gammaG_manifold[:,2], 
        label='True Geodesics', color='red')

for i in range(N_sim-1):
    ax.plot(gammaSG_manifold[i][:, 0], gammaSG_manifold[i][:, 1], 
            gammaSG_manifold[i][:, 2],  color='cyan')
ax.plot(gammaSG_manifold[N_sim-1][:, 0], gammaSG_manifold[N_sim-1][:, 1], 
        gammaSG_manifold[N_sim-1][:, 2], 
        label='Stochastic Geodesics', color='cyan')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.legend()
        
plt.tight_layout()

plt.show()

#2d plot
plt.figure(figsize=(8,6))

plt.plot(gammaG[:,0], gammaG[:,1], '-*', label='True Geodesic', color='red')

for i in range(N_sim-1):
    plt.plot(gammaSG[i][:,0], gammaSG[i][:,1], color='cyan')
plt.plot(gammaSG[N_sim-1][:,0], gammaSG[N_sim-1][:,1], label='Stochastic Geodesics', color='cyan')

plt.xlabel(r'$x^{1}$')
plt.ylabel(r'$x^{2}$')
plt.grid()
plt.legend()
plt.title('Geodesic in Z')

plt.tight_layout()

plt.show()