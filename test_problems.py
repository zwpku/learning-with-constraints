# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## This notebook defines constrained problems 

# +
import numpy as np
import torch
import math 
import torch.nn as nn
import random
import itertools 

def create_sequential_nn(layer_dims, activation=torch.nn.Tanh()):
    layers = torch.nn.Sequential()
    for i in range(len(layer_dims)-2) :
        layers.add_module(f'{i+1}',torch.nn.Linear(layer_dims[i], layer_dims[i+1])) 
        layers.add_module(f'acti {i+1}', activation)
    layers.add_module('last', torch.nn.Linear(layer_dims[-2], layer_dims[-1])) 
    return layers


# -
# ## The first problem is very simple 

class SimpleTest():   
    def __init__(self, r=1.0):
        self.r = r
        
    class Simple(nn.Module):
        def __init__(self, x):
            super(SimpleTest.Simple, self).__init__()
            self.x = torch.nn.Parameter(torch.tensor([x[0], x[1]]))
    
    def create_model(self, x):
        return self.Simple(x)
    
    def F(self, model, X=None):
        return model.x[0] + model.x[1]

    def G(self, model, X=None):
        return 0.5 * (model.x[0]**2 + (model.x[0] + model.x[1])**2 - self.r).reshape((1))
#        return 0.5 * (model.x[0]**2 - self.r).reshape((1))

# ### The second one is a Possion equation on $[0,1]^2$ with PINN loss

class PoissonEqnByPINN():
        
    def __init__(self, u_coeff=1.0, size=2000, size_each_bndry=1000, seed=100):    
        # This parameter affects the magnitude of the gradients of the solution.
        # Increasing its value makes the training more difficult.
        self.X = self.sample_data_domain(size)
        self.Y = self.sample_data_boundary(size_each_bndry)
        self.ref_u = self.ref_u_nn(u_coeff)
        self.rng = np.random.default_rng(seed)
        self.X_batch = None
        self.Y_batch = None
        
    # Reference solution
    class ref_u_nn(nn.Module):
        def __init__(self, coeff):
            super(PoissonEqnByPINN.ref_u_nn, self).__init__()
            self.coeff = coeff
        def forward(self, x):
            return (x[:,0]**2 + 1.0 * torch.sin(x[:,1] * self.coeff)).reshape((-1,1))      
    
    def create_model(self, layer_dims):
        return create_sequential_nn(layer_dims)

    def get_reference_solution(self):
        return self.ref_u

    def laplacian(self, model, x):
        u = model(x)
        u_x = torch.autograd.grad(
                u.sum(), x,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )[0]
        u_xx = 0 
        for i in range(2):
            tmp = torch.autograd.grad(
                     u_x[:,i].sum(), x,
                     retain_graph=True,
                     create_graph=True,
                     allow_unused=True
                 )[0]
            if tmp is not None:
                u_xx += tmp[:,i:i+1]
        return u_xx
    
    def sample_X_batch(self, batch_size):
        
        batch_idx_X = self.rng.integers(self.X.shape[0], size=batch_size)
        self.X_batch = torch.tensor(self.X[batch_idx_X, :], requires_grad=True).float()

    def sample_Y_batch(self, batch_size):
        
        batch_idx_Y = self.rng.integers(self.Y.shape[0], size=batch_size)
        self.Y_batch = torch.tensor(self.Y[batch_idx_Y, :]).float()  
        
    def sample_XY_batch(self, batch_size):
        self.sample_X_batch(batch_size)
        self.sample_Y_batch(batch_size)
        
    def get_X_batch(self):
        return self.X_batch
    
    def get_Y_batch(self):
        return self.Y_batch
    
    def f(self, x):   
        return self.laplacian(self.ref_u, x).detach()    

    def g(self, x):
        return self.ref_u(x).detach()

    def F(self, model):
        u_xx = self.laplacian(model, self.X_batch)
        return ((u_xx - self.f(self.X_batch))**2).mean().reshape((1))

    def G(self, model):
        u = model(self.Y_batch)
        return ((u-self.g(self.Y_batch))**2).mean().reshape((1))
    
    def sample_data_domain(self, size):
        return np.random.rand(size, 2) 

    def sample_data_boundary(self, size_each_bndry):
        # left boundary
        data_whole = np.column_stack([np.zeros(size_each_bndry), np.random.rand(size_each_bndry)]) 
        # right boundary
        data = np.column_stack([np.ones(size_each_bndry), np.random.rand(size_each_bndry)]) 
        data_whole = np.concatenate([data_whole, data])
        # top boundary
        data = np.column_stack([np.random.rand(size_each_bndry),np.ones(size_each_bndry)]) 
        data_whole = np.concatenate([data_whole, data])   
        # bottom boundary
        data = np.column_stack([np.random.rand(size_each_bndry),np.zeros(size_each_bndry)])     
        data_whole = np.concatenate([data_whole, data])

        return data_whole    

# ### The third one is an eigenvalue PDE in $\mathbb{R}^2$

class EigenPDE():
    def __init__(self, pot_V, beta, num_eig_funcs=1, eig_w=[1.0], size=2000, size_each_bndry=1000):    
        self.V = pot_V
        self.beta = beta
        self.rng = np.random.default_rng()
        self.num_eig_funcs = num_eig_funcs
        self.eig_w = eig_w
        self.X = self.sample_data_domain(size)
        self.Yb = self.sample_data_boundary(size_each_bndry)
        self.Y = [self.X, self.Yb]
        
        self.X_batch=None
        self.Yb_batch=None
        
        self._ij_list = list(itertools.combinations(range(self.num_eig_funcs), 2))
        self._num_ij_pairs = len(self._ij_list)
        
    def create_model(self, layer_dims):
        return nn.ModuleList([create_sequential_nn(layer_dims) for idx in range(self.num_eig_funcs)])
    
    def sample_X_batch(self, batch_size):
        
        batch_idx_X = self.rng.integers(self.X.shape[0], size=batch_size)
        self.X_batch = torch.tensor(self.X[batch_idx_X, :], requires_grad=True).float()
    
    def sample_Y_batch(self, batch_size):
        self.sample_X_batch(batch_size)
        self.sample_Yb_batch(batch_size)
        
    def sample_Yb_batch(self, batch_size):
        batch_idx_Yb = self.rng.integers(self.Yb.shape[0], size=batch_size)
        self.Yb_batch = torch.tensor(self.Yb[batch_idx_Yb, :]).float()  
    
    def get_X_batch(self):
        return self.X_batch
    
    def get_Y_batch(self):
        return [self.X_batch, self.Yb_batch]
    
    def F(self, model):
                    
        X = self.X_batch
        
        y = [m(X) for m in model]

        y_grad_vec = torch.stack([torch.autograd.grad(outputs=y[idx].sum(), 
                                                      inputs=X, retain_graph=True, 
                                                      create_graph=True)[0] 
                                  for idx in range(self.num_eig_funcs)], dim=2)
        
        # Mean and variance evaluated on data
        mean_list = [y[idx].mean() for idx in range(self.num_eig_funcs)]
        var_list = [(y[idx]**2).mean() - mean_list[idx]**2 for idx in range(self.num_eig_funcs)]

        # Compute Rayleigh quotients as eigenvalues
        eig_vals = torch.tensor([1.0 / self.beta * torch.mean((y_grad_vec[:,:,idx]**2).sum(dim=1)) 
                                 / var_list[idx] for idx in range(self.num_eig_funcs)])

        cvec = np.argsort(eig_vals)
        # Sort the eigenvalues 
        eig_vals = eig_vals[cvec]

        non_penalty_loss = 1.0 / self.beta * sum([self.eig_w[idx] * torch.mean((y_grad_vec[:,:,cvec[idx]]**2).sum(dim=1)) 
                                                  / (var_list[cvec[idx]]) for idx in range(self.num_eig_funcs)])

        return non_penalty_loss
            
    def G(self, model):
        X = self.X_batch
        y = [m(X) for m in model]
        
        penalty = torch.zeros(1, requires_grad=True)
        
        # Mean and variance evaluated on data
        mean_list = [y[idx].mean() for idx in range(self.num_eig_funcs)]
        var_list = [(y[idx]**2).mean() - mean_list[idx]**2 for idx in range(self.num_eig_funcs)]

        # Sum of squares of variance for each eigenfunction
        penalty = sum([(var_list[idx] - 1.0)**2 for idx in range(self.num_eig_funcs)])
                
        for idx in range(self._num_ij_pairs):
            ij = self._ij_list[idx]
            # Sum of squares of covariance between two different eigenfunctions
            penalty += ((y[ij[0]] * y[ij[1]]).mean() - mean_list[ij[0]] * mean_list[ij[1]])**2

        return penalty
    
    def sample_data_domain(self, size):
        X_uniform = np.random.rand(size, 2) 
        weights = [math.exp(-self.beta * self.V(x)) for x in X_uniform]
        return np.array(random.choices(X_uniform, weights, k=size))

    def sample_data_boundary(self, size_each_bndry):
        # left boundary
        data_whole = np.column_stack([np.zeros(size_each_bndry), np.random.rand(size_each_bndry)]) 
        # right boundary
        data = np.column_stack([np.ones(size_each_bndry), np.random.rand(size_each_bndry)]) 
        data_whole = np.concatenate([data_whole, data])
        # top boundary
        data = np.column_stack([np.random.rand(size_each_bndry),np.ones(size_each_bndry)]) 
        data_whole = np.concatenate([data_whole, data])   
        # bottom boundary
        data = np.column_stack([np.random.rand(size_each_bndry),np.zeros(size_each_bndry)])     
        data_whole = np.concatenate([data_whole, data])

        return data_whole    


