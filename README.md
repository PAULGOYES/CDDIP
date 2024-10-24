# CDDIP: Constrained Diffusion-Driven Deep Image Prior for Seismic Image Reconstruction

This repository contains the data and codes used in the paper [CDDIP: Constrained Diffusion-Driven Deep Image Prior for Seismic Image Reconstruction][(https://arxiv.org/abs/2311.10910](https://arxiv.org/abs/2407.17402)). We used Python 3.10.12 and Torch 1.13.0+cu117. We are sharing notebooks for reproducibility for the specific experiments shown in the paper. 

How to cite:
```
@misc{goyespeafiel2024cddip,
    title={CDDIP: Constrained Diffusion-Driven Deep Image Prior for Seismic Image Reconstruction},
    author={Paul Goyes-Pe\~nafiel and Ulugbek S. Kamilov and Henry Arguello},
    year={2024},
    eprint={2407.17402},
    archivePrefix={arXiv},
    primaryClass={physics.geo-ph}
}
```

Run the result for the experiment I using the pre-trained diffusion model called "DDPM_cosine_27K" : 

```
python diffDIP.py
```

Note that we are using DIP with a att-UNET ```from guided_diffusion.NetworkPaul import AttU_Net```

> [!IMPORTANT]
> This repository will be updated once the paper becomes accepted at IEEE-GRSL :octocat:

Further questions: ```goyes.yesid@gmail.com ```
