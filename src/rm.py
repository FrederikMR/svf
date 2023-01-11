#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 15:31:09 2022

@author: frederik
"""

#%% Sources

#https://bitbucket.org/stefansommer/jaxgeometry/src/main/src/Riemannian/metric.py

#%% Modules

import jax.numpy as jnp
from jax import jacfwd, jit, lax, vmap

#For double precision
from jax.config import config
config.update("jax_enable_x64", True)

import ode_integrator as oi

#%% manifold class

class riemannian_manifold(object):
    
    def __init__(self):
        
        self.G = None
        self.chris = None
        
#%% Jax functions

def rm_geometry(G = None, param_fun = None, n_steps = 100, grid = None, max_iter=100, tol=1e-05, method='euler'):
        
    def mmf(x):
        
        J = jacfwd(param_fun)(x)
        G = J.T.dot(J)
        
        return G
    
    def chris_symbols(x):
                
        G_inv = RM.G_inv(x)
        DG = RM.DG(x)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def curvature_operator(x):
        
        Dchris = RM.Dchris(x)
        chris = RM.chris(x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x):
        
        CO = RM.CO(x)
        G = RM.G(x)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2):
        
        CT = RM.CT(x)[0,1,1,0]
        G = RM.G(x)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v):
        
        def eq_geodesic(t, y):
            
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
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
    
    def bvp_geodesic(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
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
    
    def Exp(x,v):
        
        return RM.geo_ivp(x,v)[0][-1]
    
    def Log(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        dgamma = xt[:,len(x):]
        
        return dgamma[0]
    
    def pt(v0, gamma, Dgamma):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        f_fun = jit(eq_pt)
        v = oi.ode_integrator(v0, f_fun, grid = grid, method=method)
        
        return v
    
    if param_fun == None and G == None:
        raise ValueError('Both the metric matrix function and parametrization are none type. One of them has to be passed!')
    
    if grid is None:
        grid = jnp.linspace(0.0, 1.0, n_steps)
    
    RM = riemannian_manifold()
    if G is not None:
        RM.G = G
        RM.G_inv = lambda x: jnp.linalg.inv(G(x))
        RM.DG = jacfwd(G)
    else:
        RM.G = jit(mmf)
        RM.G_inv = lambda x: jnp.linalg.inv(mmf(x))  
        RM.DG = jacfwd(mmf)
        
    RM.chris = jit(chris_symbols)
    RM.Dchris = jacfwd(chris_symbols)
    
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

#%% RM for basis fun

def rm_2dbasisSG(f_fun, Df_fun, DDf_fun, DDDf_fun, N_k=5, N_l=5, n_steps = 100, grid = None, max_iter=100, tol=1e-05, method='euler'):
    
    def sum_fun(x, coef, fun):
                
        return jnp.sum(coef*vmap(lambda k: vmap(lambda l: fun(x,k,l))(l_idx))(k_idx))
    
    def mmf(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        
        g11 = val_x1*val_x1+1
        g12 = val_x1*val_x2
        g22 = val_x2*val_x2+1
        
        
        G = jnp.array([[g11, g12], [g12, g22]])
        
        return G
    
    def mmf_inv(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        
        g11 = val_x1*val_x1+1
        g12 = val_x1*val_x2
        g22 = val_x2*val_x2+1
        
        det = g11*g22-g12*g12
        G_inv = jnp.array([[g22, -g12], [-g12, g11]])/det
        
        return G_inv
    
    def Dmmf(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        val_x1x1 = sum_fun(x, coef, DDf_fun[0])
        val_x1x2 = sum_fun(x, coef, DDf_fun[1])
        val_x2x2 = sum_fun(x, coef, DDf_fun[2])
        
        Dgamma11_x1 = 2*val_x1*val_x1x1
        Dgamma11_x2 = 2*val_x1x2*val_x1
        Dgamma12_x1 = val_x1x1*val_x2+val_x1*val_x1x2
        Dgamma12_x2 = val_x1x2*val_x2+val_x1*val_x2x2
        Dgamma22_x1 = 2*val_x1x2*val_x2
        Dgamma22_x2 = 2*val_x2x2*val_x2
        
        DG = jnp.array([[[Dgamma11_x1, Dgamma11_x2],
                         [Dgamma12_x1, Dgamma12_x2]],
                        [[Dgamma12_x1, Dgamma12_x2],
                        [Dgamma22_x1, Dgamma22_x2]]])
        
        return DG
        
    def DDmmf(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        val_x1x1 = sum_fun(x, coef, DDf_fun[0])
        val_x1x2 = sum_fun(x, coef, DDf_fun[1])
        val_x2x2 = sum_fun(x, coef, DDf_fun[2])
        val_x1x1x1 = sum_fun(x, coef, DDDf_fun[0])
        val_x1x1x2 = sum_fun(x, coef, DDDf_fun[1])
        val_x2x2x1 = sum_fun(x, coef, DDDf_fun[2])
        val_x2x2x2 = sum_fun(x, coef, DDDf_fun[3])
        
        DDg11_x1x1 = 2*(val_x1x1x1*val_x1+val_x1x1*val_x1x1)
        DDg11_x1x2 = 2*(val_x1x1x2*val_x1+val_x1x1*val_x1x2)
        DDg11_x2x2 = 2*(val_x2x2x1*val_x1+val_x1x2*val_x1x2)
        DDg12_x1x1 = val_x1x1x1*val_x2+2*val_x1x1*val_x1x2+val_x1*val_x1x1x2
        DDg12_x1x2 = val_x1x1x2*val_x2+val_x1x1*val_x2x2+val_x1x2*val_x1x2+val_x1*val_x2x2x1
        DDg12_x2x2 = val_x2x2x1*val_x2+2*val_x1x2*val_x2x2+val_x1*val_x2x2x2
        DDg22_x1x1 = 2*(val_x1x1x2*val_x2+val_x1x2*val_x1x2)
        DDg22_x1x2 = 2*(val_x2x2x1*val_x2+val_x1x2*val_x2x2)
        DDg22_x2x2 = 2*(val_x2x2x2*val_x2+val_x2x2*val_x2x2)
        
        DDG = jnp.zeros((2,2,2,2))
        DDG = DDG.at[0,0,0,0].set(DDg11_x1x1)
        DDG = DDG.at[0,0,0,1].set(DDg11_x1x2)
        DDG = DDG.at[0,0,1,0].set(DDg11_x1x2)
        DDG = DDG.at[0,0,1,1].set(DDg11_x2x2)
        
        DDG = DDG.at[0,1,0,0].set(DDg12_x1x1)
        DDG = DDG.at[0,1,0,1].set(DDg12_x1x2)
        DDG = DDG.at[0,1,1,0].set(DDg12_x1x2)
        DDG = DDG.at[0,1,1,1].set(DDg12_x2x2)
        
        DDG = DDG.at[1,0,0,0].set(DDg12_x1x1)
        DDG = DDG.at[1,0,0,1].set(DDg12_x1x2)
        DDG = DDG.at[1,0,1,0].set(DDg12_x1x2)
        DDG = DDG.at[1,0,1,1].set(DDg12_x2x2)
        
        DDG = DDG.at[1,1,0,0].set(DDg22_x1x1)
        DDG = DDG.at[1,1,0,1].set(DDg22_x1x2)
        DDG = DDG.at[1,1,1,0].set(DDg22_x1x2)
        DDG = DDG.at[1,1,1,1].set(DDg22_x2x2)
        
        return DDG
    
    def detG(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        
        return (val_x1*val_x1+1)*(val_x2*val_x2+1)-(val_x1*val_x2)**2
    
    def DdetG(x, coef):
        
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        val_x1x1 = sum_fun(x, coef, DDf_fun[0])
        val_x1x2 = sum_fun(x, coef, DDf_fun[1])
        val_x2x2 = sum_fun(x, coef, DDf_fun[2])
        
        
        DdetG_x1 = 2*val_x1x1*val_x1*(1+val_x2*val_x2)+2*(1+val_x1*val_x1)*val_x1x2*val_x2-2*val_x1*val_x2*(val_x1x1*val_x2+val_x1*val_x1x2)
        DdetG_x2 = 2*val_x1x2*val_x1*(1+val_x2*val_x2)+2*(1+val_x1*val_x1)*val_x2x2*val_x2-2*val_x1*val_x2*(val_x1x2*val_x2+val_x1*val_x2x2)
        
        return jnp.array([DdetG_x1, DdetG_x2])
    
    def Dmmf_inv(x, coef):
        
        val_x1 = sum_fun(x, coef, Df_fun[0])
        val_x2 = sum_fun(x, coef, Df_fun[1])
        val_x1x1 = sum_fun(x, coef, DDf_fun[0])
        val_x1x2 = sum_fun(x, coef, DDf_fun[1])
        val_x2x2 = sum_fun(x, coef, DDf_fun[2])
        
        Ddet = DdetG(x,coef)
        det = detG(x,coef)
        det2 = det*det
        
        Dg11_inv_x1 = -Ddet[0]/det2*(1+val_x2*val_x2)+2*val_x1x2*val_x2/det
        Dg11_inv_x2 = -Ddet[1]/det2*(1+val_x2*val_x2)+2*val_x2x2*val_x2/det
        
        Dg12_inv_x1 = Ddet[0]/det2*val_x1*val_x2-(val_x1x1*val_x2+val_x1*val_x1x2)/det
        Dg12_inv_x2 = Ddet[1]/det2*val_x1*val_x2-(val_x1x2*val_x2+val_x1*val_x2x2)/det
        
        Dg22_inv_x1 = -Ddet[0]/det2*(1+val_x1*val_x1)+2*val_x1x1*val_x1/det
        Dg22_inv_x2 = -Ddet[1]/det2*(1+val_x1*val_x1)+2*val_x1x2*val_x1/det
        
        D_inv = jnp.array([[[Dg11_inv_x1, Dg11_inv_x2],
                         [Dg12_inv_x1, Dg12_inv_x2]],
                        [[Dg12_inv_x1, Dg12_inv_x2],
                        [Dg22_inv_x1, Dg22_inv_x2]]])
        
        return D_inv
    
    def Dgamma(x, coef):
        
        G_inv = RM.G_inv(x, coef)
        
        DG = Dmmf(x, coef)
        DDG = DDmmf(x, coef)
        DG_inv = Dmmf_inv(x, coef)
        
        Dchris1 = jnp.einsum('jlik, lm->mjik', DDG, G_inv) \
                    +jnp.einsum('lijk, lm->mjik', DDG, G_inv) \
                    -jnp.einsum('ijlk, lm->mjik', DDG, G_inv)
        Dchris2 = jnp.einsum('jli, lmk->mjik', DG, DG_inv) \
                    +jnp.einsum('lij, lmk->mjik', DG, DG_inv) \
                    -jnp.einsum('ijl, lmk->mjik', DG, DG_inv)
        
        return 0.5*(Dchris1+Dchris2)
    
    def chris_symbols(x, coef):
                
        G_inv = RM.G_inv(x, coef)
        DG = RM.DG(x, coef)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def curvature_operator(x, coef):
        
        Dchris = RM.Dchris(x, coef)
        chris = RM.chris(x, coef)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x, coef):
        
        CO = RM.CO(x, coef)
        G = RM.G(x, coef)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2, coef):
        
        CT = RM.CT(x, coef)[0,1,1,0]
        G = RM.G(x, coef)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v, coef):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, coef)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
            
            #G = RM.G(gamma, coef)
            #DG = RM.DG(gamma, coef)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='C')         
            
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma.reshape(1,-1), Dgamma.reshape(1,-1)).reshape(-1))))
        
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y, coef):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, coef)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
                        
            #G = RM.G(gamma, coef)
            #DG = RM.DG(gamma, coef)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='F')
                        
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Exp(x,v, coef):
        
        return RM.geo_ivp(x,v, coef)[0][-1]
    
    def Log(x,y, coef):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma, coef)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        dgamma = xt[:,len(x):]
        
        return dgamma[0]
    
    def pt(v0, gamma, Dgamma, coef):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat, coef)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        fun = jit(eq_pt)
        v = oi.ode_integrator(v0, fun, grid = grid, method=method)
        
        return v
    
    if grid is None:
        grid = jnp.linspace(0.0, 1.0, n_steps)
        
    k_idx = jnp.arange(0,N_k,1)+1
    l_idx = jnp.arange(0,N_l,1)+1
    
    RM = riemannian_manifold()
    RM.G = jit(mmf)
    RM.G_inv = jit(mmf_inv)
    RM.DG = jit(Dmmf)
    
    RM.chris = jit(chris_symbols)
    RM.Dchris = jit(Dgamma)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    RM.geo_ivp = ivp_geodesic
    RM.geo_bvp = bvp_geodesic
    RM.Exp = Exp
    RM.Log = Log
    RM.pt = jit(pt)
    
    return RM

#%% RM for basis fun for EG

def rm_2dbasisEG(f_fun, mu, Sigma, Df_fun, DDf_fun, DDDf_fun, N_k=5, N_l=5, n_steps = 100, grid = None, max_iter=100, tol=1e-05, method='euler'):
    
    def mean_sum(x, fun):
                
        return jnp.sum(mu*vmap(lambda k: vmap(lambda l: fun(x,k,l))(l_idx))(k_idx))
    
    def cov_sum(x, fun1, fun2):
                
        return jnp.sum(Sigma*vmap(lambda j: vmap(lambda i: vmap(lambda k: vmap(lambda l: fun1(x,k,l)*fun2(x,i,j))(l_idx))(k_idx))(l_idx))(k_idx))
    
    def mmf(x):
        
        cov_x1 = RM.cov_sum(x, Df_fun[0], Df_fun[0])
        cov_x2 = RM.cov_sum(x, Df_fun[1], Df_fun[1])
        cov_x1x2 = RM.cov_sum(x, Df_fun[0], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        
        g11 = cov_x1+val_x1*val_x1+1
        g12 = cov_x1x2+val_x1*val_x2
        g22 = cov_x2+val_x2*val_x2+1
        
        
        G = jnp.array([[g11, g12], [g12, g22]])
        
        return G
    
    def mmf_inv(x):
        
        cov_x1 = RM.cov_sum(x, Df_fun[0], Df_fun[0])
        cov_x2 = RM.cov_sum(x, Df_fun[1], Df_fun[1])
        cov_x1x2 = RM.cov_sum(x, Df_fun[0], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        
        g11 = cov_x1+val_x1*val_x1+1
        g12 = cov_x1x2+val_x1*val_x2
        g22 = cov_x2+val_x2*val_x2+1
        
        det = g11*g22-g12*g12
        G_inv = jnp.array([[g22, -g12], [-g12, g11]])/det
        
        return G_inv
    
    def Dmmf(x):
        
        cov_x1x1_x1 = RM.cov_sum(x, DDf_fun[0], Df_fun[0])
        cov_x1x1_x2 = RM.cov_sum(x, DDf_fun[0], Df_fun[1])
        cov_x1x2_x1 = RM.cov_sum(x, DDf_fun[1], Df_fun[0])
        cov_x1x2_x2 = RM.cov_sum(x, DDf_fun[1], Df_fun[1])
        cov_x2x2_x1 = RM.cov_sum(x, DDf_fun[2], Df_fun[0])
        cov_x2x2_x2 = RM.cov_sum(x, DDf_fun[2], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        val_x1x1 = RM.mu_sum(x, DDf_fun[0])
        val_x1x2 = RM.mu_sum(x, DDf_fun[1])
        val_x2x2 = RM.mu_sum(x, DDf_fun[2])
        
        Dgamma11_x1 = 2*(val_x1*val_x1x1+cov_x1x1_x1)
        Dgamma11_x2 = 2*(val_x1x2*val_x1+cov_x1x2_x1)
        Dgamma12_x1 = val_x1x1*val_x2+val_x1*val_x1x2+cov_x1x1_x2+cov_x1x2_x1
        Dgamma12_x2 = val_x1x2*val_x2+val_x1*val_x2x2+cov_x1x2_x2+cov_x2x2_x1
        Dgamma22_x1 = 2*(val_x1x2*val_x2+cov_x1x2_x2)
        Dgamma22_x2 = 2*(val_x2x2*val_x2+cov_x2x2_x2)
        
        DG = jnp.array([[[Dgamma11_x1, Dgamma11_x2],
                         [Dgamma12_x1, Dgamma12_x2]],
                        [[Dgamma12_x1, Dgamma12_x2],
                        [Dgamma22_x1, Dgamma22_x2]]])
        
        return DG
        
    def DDmmf(x):
        
        cov_x1x1x1_x1 = RM.cov_sum(x, DDDf_fun[0], Df_fun[0])
        cov_x1x1_x1x1 = RM.cov_sum(x, DDf_fun[0], DDf_fun[0])
        cov_x1x1_x2x2 = RM.cov_sum(x, DDf_fun[0], DDf_fun[2])
        cov_x1x1x2_x1 = RM.cov_sum(x, DDDf_fun[1], Df_fun[0])
        cov_x1x2_x1x2 = RM.cov_sum(x, DDf_fun[1], DDf_fun[1])
        cov_x1x1x1_x2 = RM.cov_sum(x, DDDf_fun[0], Df_fun[1])
        cov_x1x1_x1x2 = RM.cov_sum(x, DDf_fun[0], DDf_fun[1])
        cov_x1x1x2_x2 = RM.cov_sum(x, DDDf_fun[1], Df_fun[1])
        cov_x2x2x1_x1 = RM.cov_sum(x, DDDf_fun[2], Df_fun[0])
        cov_x2x2x1_x2 = RM.cov_sum(x, DDDf_fun[2], Df_fun[1])
        cov_x1x2_x2x2 = RM.cov_sum(x, DDf_fun[1], DDf_fun[2])
        cov_x2x2x2_x1 = RM.cov_sum(x, DDDf_fun[3], Df_fun[0])
        cov_x2x2x2_x2 = RM.cov_sum(x, DDDf_fun[3], Df_fun[1])
        cov_x2x2_x2x2 = RM.cov_sum(x, DDf_fun[2], DDf_fun[2])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        val_x1x1 = RM.mu_sum(x, DDf_fun[0])
        val_x1x2 = RM.mu_sum(x, DDf_fun[1])
        val_x2x2 = RM.mu_sum(x, DDf_fun[2])
        val_x1x1x1 = RM.mu_sum(x, DDDf_fun[0])
        val_x1x1x2 = RM.mu_sum(x, DDDf_fun[1])
        val_x2x2x1 = RM.mu_sum(x, DDDf_fun[2])
        val_x2x2x2 = RM.mu_sum(x, DDDf_fun[3])
        
        DDg11_x1x1 = 2*(val_x1x1x1*val_x1+val_x1x1*val_x1x1+cov_x1x1x1_x1+cov_x1x1_x1x1)
        DDg11_x1x2 = 2*(val_x1x1x2*val_x1+val_x1x1*val_x1x2+cov_x1x1x2_x1+cov_x1x1_x1x2)
        DDg11_x2x2 = 2*(val_x2x2x1*val_x1+val_x1x2*val_x1x2+cov_x2x2x1_x1+cov_x1x2_x1x2)
        DDg12_x1x1 = val_x1x1x1*val_x2+2*val_x1x1*val_x1x2+val_x1*val_x1x1x2+cov_x1x1x1_x2+2*cov_x1x1_x1x2+cov_x1x1x2_x1
        DDg12_x1x2 = val_x1x1x2*val_x2+val_x1x1*val_x2x2+val_x1x2*val_x1x2+val_x1*val_x2x2x1+cov_x1x1x2_x2+cov_x1x1_x2x2+cov_x1x2_x1x2+cov_x2x2x1_x1
        DDg12_x2x2 = val_x2x2x1*val_x2+2*val_x1x2*val_x2x2+val_x1*val_x2x2x2+cov_x2x2x1_x2+2*cov_x1x2_x2x2+cov_x2x2x2_x1
        DDg22_x1x1 = 2*(val_x1x1x2*val_x2+val_x1x2*val_x1x2+cov_x1x1x2_x2+cov_x1x2_x1x2)
        DDg22_x1x2 = 2*(val_x2x2x1*val_x2+val_x1x2*val_x2x2+cov_x2x2x1_x2+cov_x1x2_x2x2)
        DDg22_x2x2 = 2*(val_x2x2x2*val_x2+val_x2x2*val_x2x2+cov_x2x2x2_x2+cov_x2x2_x2x2)
        
        DDG = jnp.zeros((2,2,2,2))
        DDG = DDG.at[0,0,0,0].set(DDg11_x1x1)
        DDG = DDG.at[0,0,0,1].set(DDg11_x1x2)
        DDG = DDG.at[0,0,1,0].set(DDg11_x1x2)
        DDG = DDG.at[0,0,1,1].set(DDg11_x2x2)
        
        DDG = DDG.at[0,1,0,0].set(DDg12_x1x1)
        DDG = DDG.at[0,1,0,1].set(DDg12_x1x2)
        DDG = DDG.at[0,1,1,0].set(DDg12_x1x2)
        DDG = DDG.at[0,1,1,1].set(DDg12_x2x2)
        
        DDG = DDG.at[1,0,0,0].set(DDg12_x1x1)
        DDG = DDG.at[1,0,0,1].set(DDg12_x1x2)
        DDG = DDG.at[1,0,1,0].set(DDg12_x1x2)
        DDG = DDG.at[1,0,1,1].set(DDg12_x2x2)
        
        DDG = DDG.at[1,1,0,0].set(DDg22_x1x1)
        DDG = DDG.at[1,1,0,1].set(DDg22_x1x2)
        DDG = DDG.at[1,1,1,0].set(DDg22_x1x2)
        DDG = DDG.at[1,1,1,1].set(DDg22_x2x2)
        
        return DDG
    
    def detG(x):
        
        cov_x1 = RM.cov_sum(x, Df_fun[0], Df_fun[0])
        cov_x2 = RM.cov_sum(x, Df_fun[1], Df_fun[1])
        cov_x1x2 = RM.cov_sum(x, Df_fun[0], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        
        return (val_x1*val_x1+cov_x1+1)*(val_x2*val_x2+cov_x2+1)-(val_x1*val_x2+cov_x1x2)**2
    
    def DdetG(x):
        
        cov_x1 = RM.cov_sum(x, Df_fun[0], Df_fun[0])
        cov_x2 = RM.cov_sum(x, Df_fun[1], Df_fun[1])
        cov_x1x2 = RM.cov_sum(x, Df_fun[0], Df_fun[1])
        
        cov_x1x1_x1 = RM.cov_sum(x, DDf_fun[0], Df_fun[0])
        cov_x1x1_x2 = RM.cov_sum(x, DDf_fun[0], Df_fun[1])
        cov_x1x2_x1 = RM.cov_sum(x, DDf_fun[1], Df_fun[0])
        cov_x1x2_x2 = RM.cov_sum(x, DDf_fun[1], Df_fun[1])
        cov_x2x2_x1 = RM.cov_sum(x, DDf_fun[2], Df_fun[0])
        cov_x2x2_x2 = RM.cov_sum(x, DDf_fun[2], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        val_x1x1 = RM.mu_sum(x, DDf_fun[0])
        val_x1x2 = RM.mu_sum(x, DDf_fun[1])
        val_x2x2 = RM.mu_sum(x, DDf_fun[2])
        
        
        DdetG_x1 = 2*(val_x1x1*val_x1+cov_x1x1_x1)*(1+val_x2*val_x2+cov_x2)+2*(1+val_x1*val_x1+cov_x1)*(val_x1x2*val_x2+cov_x1x2_x2)-2*(val_x1*val_x2+cov_x1x2)*(val_x1x1*val_x2+val_x1*val_x1x2+cov_x1x1_x2+cov_x1x2_x1)
        DdetG_x2 = 2*(val_x1x2*val_x1+cov_x1x2_x1)*(1+val_x2*val_x2+cov_x2)+2*(1+val_x1*val_x1+cov_x1)*(val_x2x2*val_x2+cov_x2x2_x2)-2*(val_x1*val_x2+cov_x1x2)*(val_x1x2*val_x2+val_x1*val_x2x2+cov_x1x2_x2+cov_x2x2_x1)
        
        return jnp.array([DdetG_x1, DdetG_x2])
    
    def Dmmf_inv(x):
        
        cov_x1 = RM.cov_sum(x, Df_fun[0], Df_fun[0])
        cov_x2 = RM.cov_sum(x, Df_fun[1], Df_fun[1])
        cov_x1x2 = RM.cov_sum(x, Df_fun[0], Df_fun[1])
        
        cov_x1x1_x1 = RM.cov_sum(x, DDf_fun[0], Df_fun[0])
        cov_x1x1_x2 = RM.cov_sum(x, DDf_fun[0], Df_fun[1])
        cov_x1x2_x1 = RM.cov_sum(x, DDf_fun[1], Df_fun[0])
        cov_x1x2_x2 = RM.cov_sum(x, DDf_fun[1], Df_fun[1])
        cov_x2x2_x1 = RM.cov_sum(x, DDf_fun[2], Df_fun[0])
        cov_x2x2_x2 = RM.cov_sum(x, DDf_fun[2], Df_fun[1])
        
        val_x1 = RM.mu_sum(x, Df_fun[0])
        val_x2 = RM.mu_sum(x, Df_fun[1])
        val_x1x1 = RM.mu_sum(x, DDf_fun[0])
        val_x1x2 = RM.mu_sum(x, DDf_fun[1])
        val_x2x2 = RM.mu_sum(x, DDf_fun[2])
        
        Ddet = DdetG(x)
        det = detG(x)
        det2 = det*det
        
        Dg11_inv_x1 = -Ddet[0]/det2*(1+val_x2*val_x2+cov_x2)+2*(val_x1x2*val_x2+cov_x1x2_x2)/det
        Dg11_inv_x2 = -Ddet[1]/det2*(1+val_x2*val_x2+cov_x2)+2*(val_x2x2*val_x2+cov_x2x2_x2)/det
        
        Dg12_inv_x1 = Ddet[0]/det2*(val_x1*val_x2+cov_x1x2)-(val_x1x1*val_x2+val_x1*val_x1x2+cov_x1x1_x2+cov_x1x2_x1)/det
        Dg12_inv_x2 = Ddet[1]/det2*(val_x1*val_x2+cov_x1x2)-(val_x1x2*val_x2+val_x1*val_x2x2+cov_x1x2_x2+cov_x2x2_x1)/det
        
        Dg22_inv_x1 = -Ddet[0]/det2*(1+val_x1*val_x1+cov_x1)+2*(val_x1x1*val_x1+cov_x1x1_x1)/det
        Dg22_inv_x2 = -Ddet[1]/det2*(1+val_x1*val_x1+cov_x1)+2*(val_x1x2*val_x1+cov_x1x2_x1)/det
        
        D_inv = jnp.array([[[Dg11_inv_x1, Dg11_inv_x2],
                         [Dg12_inv_x1, Dg12_inv_x2]],
                        [[Dg12_inv_x1, Dg12_inv_x2],
                        [Dg22_inv_x1, Dg22_inv_x2]]])
        
        return D_inv
    
    def Dgamma(x):
        
        G_inv = RM.G_inv(x)
        
        DG = Dmmf(x)
        DDG = DDmmf(x)
        DG_inv = Dmmf_inv(x)
        
        Dchris1 = jnp.einsum('jlik, lm->mjik', DDG, G_inv) \
                    +jnp.einsum('lijk, lm->mjik', DDG, G_inv) \
                    -jnp.einsum('ijlk, lm->mjik', DDG, G_inv)
        Dchris2 = jnp.einsum('jli, lmk->mjik', DG, DG_inv) \
                    +jnp.einsum('lij, lmk->mjik', DG, DG_inv) \
                    -jnp.einsum('ijl, lmk->mjik', DG, DG_inv)
        
        return 0.5*(Dchris1+Dchris2)
    
    def chris_symbols(x):
                
        G_inv = RM.G_inv(x)
        DG = RM.DG(x)
        
        chris = 0.5*(jnp.einsum('jli, lm->mji', DG, G_inv) \
                     +jnp.einsum('lij, lm->mji', DG, G_inv) \
                     -jnp.einsum('ijl, lm->mji', DG, G_inv))
        
        return chris
    
    def curvature_operator(x):
        
        Dchris = RM.Dchris(x)
        chris = RM.chris(x)
        
        R = jnp.einsum('mjki->mijk', Dchris) \
            -jnp.einsum('mikj->mijk', Dchris) \
            +jnp.einsum('sjk,mis->mijk', chris, chris) \
            -jnp.einsum('sik, mjs->mijk', chris, chris)
        
        return R
    
    def curvature_tensor(x):
        
        CO = RM.CO(x)
        G = RM.G(x)
        
        CT = jnp.einsum('sijk, sm -> ijkm', CO, G)
        
        return CT
    
    def sectional_curvature(x, e1, e2):
        
        CT = RM.CT(x)[0,1,1,0]
        G = RM.G(x)
        
        return CT/(e1.dot(G).dot(e1)*e2.dot(G).dot(e2)-(e1.dot(G).dot(e2))**2)
    
    def ivp_geodesic(x,v):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
            
            #G = RM.G(gamma, coef)
            #DG = RM.DG(gamma, coef)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='C')         
            
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma.reshape(1,-1), Dgamma.reshape(1,-1)).reshape(-1))))
        
        N = len(x)
        y0 = jnp.concatenate((x, v), axis=0)
        f_fun = jit(eq_geodesic)
        y = oi.ode_integrator(y0, f_fun, grid = grid, method=method)
        gamma = y[:,0:len(x)]
        Dgamma = y[:,len(x):]
        
        return gamma, Dgamma
    
    def bvp_geodesic(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
            
            #gamma = y[0:N]
            #Dgamma = y[N:]
                        
            #G = RM.G(gamma, coef)
            #DG = RM.DG(gamma, coef)
            
            #DG = jnp.einsum('ijk->kij', DG).reshape(N,-1, order='F')
                        
            #return jnp.concatenate((Dgamma, -0.5*jnp.linalg.solve(G, DG).dot(jnp.kron(Dgamma, Dgamma))))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        gamma = xt[:,0:len(x)]
        Dgamma = xt[:,len(x):]
        
        return gamma, Dgamma
    
    def Exp(x,v):
        
        return RM.geo_ivp(x,v)[0][-1]
    
    def Log(x,y):
        
        def eq_geodesic(t, y):
            
            gamma = y[0:N]
            Dgamma = y[N:]
            
            chris = RM.chris(gamma)
            
            return jnp.concatenate((Dgamma, -jnp.einsum('i,j,kij->k', Dgamma, Dgamma, chris)))
        
        N = len(x)
        f_fun = jit(eq_geodesic)
        xt = oi.bvp_solver(jnp.zeros_like(x), x, y, f_fun, grid = grid, max_iter=max_iter, tol=tol, method=method)
        dgamma = xt[:,len(x):]
        
        return dgamma[0]
    
    def pt(v0, gamma, Dgamma):
        
        def eq_pt(t:jnp.ndarray, v:jnp.ndarray)->jnp.ndarray:
            
            idx = jnp.argmin(jnp.abs(grid-t))
            gammat = gamma[idx]
            Dgammat = Dgamma[idx]
            
            chris = RM.chris(gammat)
            
            return -jnp.einsum('i,j,kij->k', v, Dgammat, chris)
        
        fun = jit(eq_pt)
        v = oi.ode_integrator(v0, fun, grid = grid, method=method)
        
        return v
    
    if grid is None:
        grid = jnp.linspace(0.0, 1.0, n_steps)
        
    k_idx = jnp.arange(0,N_k,1)+1
    l_idx = jnp.arange(0,N_l,1)+1
    
    RM = riemannian_manifold()
    RM.mu_sum = mean_sum
    RM.cov_sum = cov_sum
    RM.G = jit(mmf)
    RM.G_inv = jit(mmf_inv)
    RM.DG = jit(Dmmf)
    
    RM.chris = jit(chris_symbols)
    RM.Dchris = jit(Dgamma)
    
    RM.CO = jit(curvature_operator)
    RM.CT = jit(curvature_tensor)
    RM.SC = jit(sectional_curvature)
    RM.geo_ivp = ivp_geodesic
    RM.geo_bvp = bvp_geodesic
    RM.Exp = Exp
    RM.Log = Log
    RM.pt = jit(pt)
    
    return RM



