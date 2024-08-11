import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
from guided_diffusion.script_util import create_model
import logging
from torch.utils.tensorboard import SummaryWriter
from ddpm import Diffusion
from guided_diffusion.NetworkPaul import AttU_Net

ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model = create_model(image_size=128,num_channels=64,num_res_blocks=3)
model = model.to(device)

#models/DDPM_cosine/ckpt.pt
#

ckpt = torch.load("models/DDPM_cosine_27K/ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=128, device=device)
#x = diffusion.sample_single(model) # sample single image
x = diffusion.sample(model,n=100)



#x = diffusion.resample_single(model,iter_num=1000,skip_type='uniform')
#plot_images(x)
save_images(x, "results/sampled_images/imgs_cosine.jpg")


