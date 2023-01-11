#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:35:48 2023

@author: frederik
"""

#%% Sources

#https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry

#%% Modules

import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

import ode_integrator as oi

#%% manifold class

class svfrm(object):
    
    def __init__(self):
        
        self.G = None
        self.chris = None
        
#%% Jax functions

def gp_svfrm(mu_fun, sigma_fun, D, G = None, param_fun = None, n_steps = 100, 
             grid = None, max_iter=100, tol=1e-05, method='euler'):
        
    def mmf(x):
        
        J = jacfwd(param_fun)(x)
        G = J.T.dot(J)
        
        return G
    
    def chris_symbols(x, eta):
                
        G_inv = RM.G_inv(x)
        DG = RM.DG(x)
        eps = RM.mu(x)+RM.sigma(x)*eta
        deps = RM.Dmu(x)+RM.Dsigma(x)*eta
        
        chris = (eps**2)*0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
            
        chris += eps*deps*delta_mat
        
        return chris
    
    def curvature_operator(x, eta):
        
        Dchris = RM.Dchris(x, 0.0) #(..., eta)
        chris = RM.chris(x, 0.0) #(..., eta)
        eps = RM.mu(x)+RM.sigma(x)*eta
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return (eps**3)*R
    
    def curvature_tensor(x, eta):
        
        CO = RM.CO(x, eta)
        G = RM.G(x)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2, eta):
        
        CT = RM.CT(x, eta)[0,1,1,0]
        G = RM.G(x)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v, eta):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, eta)
            eps = RM.mu(x)+RM.sigma(x)*eta
            
            return jnp.concatenate((Dgamma, 
                                    -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)/(eps**2+1e-10)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
            
            #G = RM.G(gamma)
            #DG = RM.DG(gamma)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='F')
                        
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
                    
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y, eta):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, eta)
            eps = RM.mu(gamma)+RM.sigma(gamma)*eta
            
            return jnp.concatenate((Dgamma, 
                                    -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)/(eps**2+1e-10)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
                        
            #G = RM.G(gamma)
            #DG = RM.DG(gamma)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='F')
                        
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Exp(x,v, eta):
        
        return RM.geo_ivp(x,v, eta)[0][-1]
    
    def Log(x,y, eta):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, eta)
            eps = RM.mu(x)+RM.sigma(x)*eta
            
            return jnp.concatenate((Dgamma, 
                                    -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)/(eps**2+1e-10)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        dgamma = xt[:,len(x):]
        
        return dgamma[0]
    
    def pt(v0, gamma, Dgamma, eta):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            eps = RM.mu(gammat)+RM.sigma(gammat)*eta
            
            chris = RM.chris(gammat, eta)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)/(eps**2+1e-10)
        
        f_fun = jit(eq_pt)
        v = oi.ode_integrator(v0, f_fun, grid = grid, method=method)
        
        return v
    
    if param_fun == None and G == None:
        raise ValueError('Both the metric matrix function and parametrization are none type. One of them has to be passed!')
    
    if grid is None:
        grid = jnp.linspace(0.0, 1.0, n_steps)
    
    idx = jnp.arange(D)
    delta_mat = jnp.zeros((D,D,D))
    delta_mat = delta_mat.at[idx, idx, :].set(1.0)
    
    RM = svfrm()
    if G is not None:
        RM.G = G
        RM.G_inv = lambda x: jnp.linalg.inv(G(x))
        RM.DG = jacfwd(G)
    else:
        RM.G = jit(mmf)
        RM.G_inv = lambda x: jnp.linalg.inv(mmf(x))  
        RM.DG = jacfwd(mmf)
        
    RM.chris = jit(chris_symbols)
    RM.Dchris = jacfwd(chris_symbols, argnums=0)
    RM.mu = jit(mu_fun)
    RM.Dmu = jacfwd(mu_fun)
    RM.sigma = jit(sigma_fun)
    RM.Dsigma = jacfwd(sigma_fun)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    RM.geo_ivp = jit(ivp_geodesic)
    RM.geo_bvp = bvp_geodesic
    RM.Exp = Exp
    RM.Log = Log
    RM.pt = jit(pt)
    
    return RM

#%% RM statistics

def km_gs(RM, X, mu0 = None, tau = 1.0, n_steps = 100): #Karcher mean, gradient search

    def step(mu, idx):
        
        Log_sum = jnp.sum(vmap(lambda x: RM.Log(mu,x))(X), axis=0)
        delta_mu = tauN*Log_sum
        mu = RM.Exp(mu, delta_mu)
        
        return mu, mu
    
    N,d = X.shape #N: number of data, d: dimension of manifold
    tauN = tau/N
    
    if mu0 is None:
        mu0 = X[0]
        
    mu, _ = lax.scan(step, init=mu0, xs=jnp.zeros(n_steps))
            
    return mu

def pga(RM, X, acceptance_rate = 0.95, normalise=False, tau = 1.0, n_steps = 100):
    
    N,d = X.shape #N: number of data, d: dimension of manifold
    mu = km_gs(RM, X, tau = 1.0, n_steps = 100)
    u = vmap(lambda x: RM.Log(mu,x))(X)
    S = u.T
    
    if normalise:
        mu_norm = S.mean(axis=0, keepdims=True)
        std = S.std(axis=0, keepdims=True)
        X_norm = (S-mu_norm)/std
    else:
        X_norm = S
    
    U,S,V = jnp.linalg.svd(X_norm,full_matrices=False)
    
    rho = jnp.cumsum((S*S) / (S*S).sum(), axis=0) 
    n = 1+len(rho[rho<acceptance_rate])
 
    U = U[:,:n]
    rho = rho[:n]
    V = V[:,:n]
    # Project the centered data onto principal component space
    Z = X_norm @ V
    
    pga_component = vmap(lambda v: RM.Exp(mu, v))(V)
    pga_proj = vmap(lambda v: RM.Exp(mu, v))(Z)
    
    return rho, U, V, Z, pga_component, pga_proj