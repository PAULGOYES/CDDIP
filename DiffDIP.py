import os
from typing import Any
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
import cv2
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm import trange

from torch import optim
from utils import *
from modules import UNet
from guided_diffusion.script_util import create_model
import logging
from torch.utils.tensorboard import SummaryWriter
from ddpm import Diffusion
from guided_diffusion.NetworkPaul import AttU_Net
import utils_image as util
from guided_diffusion.solverDIP import *

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
model = create_model(image_size=128,num_channels=64,num_res_blocks=3)
model = model.to(device)
PSNR = PeakSignalNoiseRatio().to(device)
SSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

ckpt = torch.load("models/DDPM_cosine_27K/ckpt.pt")
model.load_state_dict(ckpt)
diffusion = Diffusion(img_size=128, device=device)
#x = diffusion.sample_single(model)


model_DPI = AttU_Net(img_ch=3,output_ch=3)
#exp 1: 11dbf86bab

#pathimg = 'dataset/test_data2/Alaska/patch_39.jpg'
pathimg = 'dataset/test/11dbf86bab.png'
img_H = util.imread_uint(pathimg, n_channels=3)/255.
img_H = img_H.transpose(2,0,1)[np.newaxis,:,:,:]
img_H: int | Any=img_H*2-1
img_H=torch.as_tensor(img_H.copy())

mask = torch.ones(img_H.shape)
traces = np.arange(128)[5:-5]
np.random.seed(12345) #only for reproducibility experiment 1
#np.random.seed(1234) #only for reproducibility experiment 2
#np.random.seed(12345) #only for reproducibility experiment 3

indx = np.random.permutation(traces)[:int(128*0.5)]
mask[:,:,:,indx]=0
##cv2.imwrite("results/diffDIPexp3/mask.jpg", mask[0,0,...].clone().detach().cpu().numpy()*255)
##cv2.imwrite("results/diffDIPexp3/target.jpg", util.imread_uint(pathimg, n_channels=3))

y =img_H*mask
#cv2.imwrite("results/diffDIPexp2/measurement.jpg", util.single2uint(y[0,0,...].clone().detach()))

measurement = (y.clone().clamp(-1, 1) + 1) / 2
measurement = (measurement * mask*255).type(torch.uint8)
##save_images(measurement,"results/diffDIPexp3/measurement.jpg")
np.random.seed()
##########################
noise_steps=1000
iter_num=25
skip_type='uniform'
skip = noise_steps // iter_num  # skip interval
if skip_type == 'uniform':
    seq: list[int] = [i * skip for i in range(iter_num)]
    if skip > 1:
        seq.append(noise_steps - 1)
seq = seq[::-1]
progress_seq = seq[::(len(seq) // 10)]


###############################
#noise schedule from diffusion
beta = diffusion.prepare_noise_schedule(schedule_name='cosine').to(device)
alpha = 1-beta
alpha_hat = torch.cumprod(alpha, dim=0)
##################################


x = torch.randn((1, 3, 128, 128)).to(device)



progress_img = []
progress_xdip= []
pbar =  trange(len(seq))
for i in pbar:
    #logging.info(f"Sampling Diff step {seq[i]} ")

    x_hat = diffusion.resample_single(model, seq[i], x)

    x,cost = trainDPI(model_DPI,mask,y,device,epochs=50,input_x=x_hat.clamp(-1, 1))
    pbar.set_postfix(MSEdip=cost[-1],DiffStep=seq[i])

    if seq[i]>1:
        x= torch.sqrt(alpha_hat[seq[i+1]])*x + torch.sqrt(1 - alpha_hat[seq[i+1]])*torch.randn_like(x,device=device)

    if seq[i] in progress_seq:
        #logging.info(f"saving progress ")
        #print('saving....')
        progress_img.append(x_hat)
        progress_xdip.append(x)



result = torch.cat(progress_img, dim=0)
result = (result.clamp(-1, 1) + 1) / 2
result = (result * 255).type(torch.uint8)

resultdip = torch.cat(progress_xdip, dim=0)
resultdip = (resultdip.clamp(-1, 1) + 1) / 2
resultdip = (resultdip * 255).type(torch.uint8)

save_images(resultdip[-1], "results/diffDIP/recover.jpg")
#save_images(result, "results/diffDIPexp3/progress_xhat.jpg",nrow=len(progress_seq))
#save_images(resultdip, "results/diffDIPexp3/progress_xdip.jpg",nrow=len(progress_seq))

print(progress_seq)

psnr0=PSNR(x_hat.clamp(-1, 1) ,img_H.to(device, dtype=torch.float32))
ssim0=SSIM(x_hat.clamp(-1, 1) ,img_H.to(device, dtype=torch.float32))
print('PSNR:', psnr0)
print('SSIM:', ssim0)

