#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 15:32:29 2023

@author: julia
"""



import torch
import torchvision.datasets as datasets
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder #imadethis
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset 
import matplotlib.pyplot as plt
import numpy as np
#%%
def inference(digit,num_examples=1):
    images = []
    idx = 0
    for x, y in dataset:
        if y==idx:
            images.append(x)
            idx += 1
        if y==10:
            break
        
    encodings_digit = [] #gets mu and sigma for each digit
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1,28*28))
        encodings_digit.append((mu,sigma))
 
    mu, sigma = encodings_digit[digit]
    for example in range(num_examples):
        epsilon = torch.randn_like(sigma)
        z = mu+sigma*epsilon
        out = model.decode(z)
        out = out.view(-1,1,28,28)
        plt.figure(figsize=(10,10))
        figura = np.asarray(out[0,0,:,:].detach())
        plt.imshow(figura)
        plt.show
#%%
def normal_dist(X, mean, sd):
    prob_density = (1/(sd*np.sqrt(2*np.pi)))*np.exp(-(X-mean)**2/(2*sd**2))
    return prob_density
#%%
N = 10000
mu_data = 2
sigma_data = 1/3
x_data = []
for i in range(N):
    data = np.reshape(np.random.normal(mu_data,sigma_data,1),(-1,))
    data = data.astype('float32')
    data = torch.Tensor(data)
    x_data.append(data)

x_data = torch.Tensor(x_data)

    

'''
x2 = np.random.normal(3,1,N)
x3 = np.random.normal(-4,3,N)
mu = 0
sigma = 1/3
plt.plot(bin_edges, 1/(sigma * np.sqrt(2 * np.pi)) *
               np.exp( - (bin_edges - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
hist,bin_edges = np.histogram(x2, bins=30, density=True)
plt.plot(bin_edges[:-1],hist)
hist,bin_edges = np.histogram(x3, bins=30, density=True)
plt.plot(bin_edges[:-1],hist)
'''

hist,bin_edges = np.histogram(x_data, bins=30, density=True)
plt.plot(bin_edges[:-1],hist)

x_true = np.linspace(x_data.min(),x_data.max(),500)
y_true = normal_dist(x_true,mu_data,sigma_data) #+ 0.3*normal_dist(x,3,1) + 0.2*normal_dist(x,-4,3)
plt.plot(x_true,y_true)
#%%
INPUT_DIM = 1
Z_DIM = 1 #larger network
H_DIM = 200 #more compression

NUM_EPOCHS = 50
BATCH_SIZE = 64
LR_RATE = 3e-4 #the karpathy constant (?)


train_loader = DataLoader(dataset=x_data,
                          batch_size=BATCH_SIZE, 
                          shuffle=True)

model = VariationalAutoEncoder(INPUT_DIM, H_DIM, Z_DIM)#.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
#loss_fn = F.binary_cross_entropy_with_logits() # maybe mse .nn.functional.binary_cross_entropy_with_logit
loss_fn = nn.BCELoss(reduction='sum') # maybe mse .nn.functional.binary_cross_entropy_with_logit
#%%
#for training
for epoch in range(NUM_EPOCHS):
    
    loop = tqdm(enumerate(train_loader))
    for i, x in loop:
        #torch.Size([64, 784])    
        x = x.view(x.shape[0],INPUT_DIM)#.to(DEVICE)
        #print(x.shape)
        x_reconstructed, mu, sigma = model(x)
        print('mu =', mu, ', sigma =', sigma)
        recon_loss = loss_fn(x_reconstructed, x)
        kl_div = - torch.sum(1 + torch.log(sigma.pow(2)) - 
                             mu.pow(2) - 
                             sigma.pow(2)) #pushes towards gaussian
        
        
        loss = recon_loss + kl_div
        optimizer.zero_grad() #zeroes the grandient buffers to all parameters of the nn
        loss.backward() #the whole graph is differentiated w.r.t. the loss, will accum .grad for all variables
        optimizer.step() #updates parameters of the network
    
        loop.set_postfix(loss=loss.item())
        
#%%
mu_gen = torch.Tensor(np.arange(-0.5, 0.5, 0.1,dtype='float32'))
sigma_gen = torch.Tensor(np.arange(0.8,1.2,0.1,dtype='float32'))   
     
#%%
i=0
j=0
for i in enumerate(mu_gen):
    for j in enumerate(sigma_gen):
        epsilon = torch.randn_like(sigma_gen[j])
        z = torch.reshape(mu_gen[i]+sigma_gen[j]*epsilon,(1,))
        out = model.decode(z)
#idx=1    
#inference(idx,num_examples=1)
        
    