import os.path
import cv2
import logging
import tqdm
from tqdm import trange

import numpy as np
import torch.nn as nn
from collections import OrderedDict

import torch
import torch.nn.functional as F

from tqdm import tqdm

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def trainDPI(model,mask,y,device,epochs=100,input_x=None):
    loss = []
    optimiser = torch.optim.AdamW(model.parameters(),lr=1e-3)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser,patience=10,verbose=True)

    model = model.to(device=device)


    np.random.seed(seed=12345) # for reproducibility
    noise = torch.rand(1,3,128,128)
    noise = noise.to(device=device)
    if input_x is not None:
        input_x = input_x.to(device, dtype=torch.float32)
        noise = input_x#+noise


    
    

    mask = mask.to(device=device,dtype=torch.float32)
    y = y.to(device=device,dtype=torch.float32)


    #pbar = trange(epochs)
    for epoch in range(epochs):
        model.train()
        
        yi = mask*y
        out = model(noise)
        
        
        ##calculate loss
        lmb = 0.1
        cost = F.mse_loss(input=mask*out, target=yi)
        #costv = F.mse_loss(input=mask[:,1,:,:]*out[:,1,:,:], target=yi[:,1,:,:])
        #costTV = TVLoss(TVLoss_weight=.01)

        

        ##backprop
        cost.backward()
        optimiser.step()
        #pbar.set_postfix(MSE=cost.item())
        optimiser.zero_grad()
        loss.append(cost.item())

    model.eval()
    with torch.no_grad():
        Out=model(noise)
    return Out/torch.max(torch.abs(Out[:])),loss

