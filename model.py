#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 10:12:52 2023

@author: julia
"""
import torch
from torch import nn
#take input img
#take to hidden dim
#output mean and std dev
#do the parametrization trick
#send it to decoder
#get output img (should be same as input)

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        #encoder
        super().__init__()
        self.img_2hid = nn.Linear(input_dim, h_dim) #inear layer
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)
        
        #decoder
        self.z_2hid = nn.Linear(z_dim,h_dim)
        self.hid_2img = nn.Linear(h_dim,input_dim)
        
        self.relu = nn.ReLU()
        
    def encode(self,x):
        #q_phi(z|x)
        #print(self.img_2hid.weight.dtype)
        h = self.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)
        return mu, sigma
    
    def decode(self,z):
        #p_theta(x|z)
        h = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(h))
    
    def forward(self,x):
        mu, sigma = self.encode(x)
        epsilon = torch.randn_like(sigma)
        z_reparametrized = mu+sigma*epsilon
        x_reconstructed = self.decode(z_reparametrized)    
        return x_reconstructed, mu, sigma
    
'''
if __name__ == "__main__":
    x = torch.randn(4,784)
    input_dim = 784
    vae  = VariationalAutoEncoder(input_dim)
    x_reconstructed, mu, sigma = vae(x)
    print(x_reconstructed.shape)
    print(mu.shape)
    print(sigma.shape)
    
    
#    print(vae(x).shape)
'''