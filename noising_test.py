import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
args.batch_size = 1  # 5
args.image_size = 128
args.dataset_path = "seismic"

dataloader = get_data(args)

diff = Diffusion(img_size=128,device="cpu")

image = next(iter(dataloader))[0]
times = [50, 100, 150, 200, 300, 600, 700, 999]
t = torch.Tensor( times[::-1]).long()

noised_image, _ = diff.noise_images(image, t)
save_image(noised_image.add(1).mul(0.5), "noise.jpg")
