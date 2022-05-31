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

# +
import numpy as np
import torch
import math 
import torch.nn as nn
import random

class SimpleTest():   
    def __init__(self, r=1.0):
        self.r = r
        
    class Simple(nn.Module):
        def __init__(self, x):
            super(SimpleTest.Simple, self).__init__()
            self.x = torch.nn.Parameter(torch.tensor([x[0], x[1]]))
    
    def create_model(self, x):
        return self.Simple(x)
    
    def f(self, model, X=None):
        return model.x[0] + model.x[1]

    def g(self, model, X=None):
        return 0.5 * (model.x[0]**2 + (model.x[0] + model.x[1])**2 - self.r).reshape((1))
    
     
class PoissonEqnByPINN():
        
    def __init__(self, u_coeff=1.0, size=2000, size_each_bndry=1000):    
        # This parameter affects the magnitude of the gradients of the solution.
        # Increasing its value makes the training more difficult.
        self.X = self.sample_data_domain(size)
        self.Y = self.sample_data_boundary(size_each_bndry)
        self.ref_u = self.ref_u_nn(u_coeff)

    # Reference solution
    class ref_u_nn(nn.Module):
        def __init__(self, coeff):
            super(PoissonEqnByPINN.ref_u_nn, self).__init__()
            self.coeff = coeff
        def forward(self, x):
            return (x[:,0]**2 + 1.0 * torch.sin(x[:,1] * self.coeff)).reshape((-1,1))      

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
    
    def get_X(self):
        return self.X
    
    def get_Y(self):
        return self.Y
    
    def f(self, x):   
        return self.laplacian(self.ref_u, x).detach()    

    def g(self, x):
        return self.ref_u(x).detach()

    def F(self, model, x):
        u_xx = self.laplacian(model, x)
        return ((u_xx - self.f(x))**2).mean().reshape((1))

    def G(self, model, x):
        u = model(x)
        return ((u-self.g(x))**2).mean().reshape((1))
    
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
# -


