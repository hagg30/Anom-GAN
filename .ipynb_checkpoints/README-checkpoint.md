# Anomaly Mask GAN

Implementation of AnoM-GAN: A Novel Unsupervised Anomaly Localization with GAN Manifold Mask Model in PyTorch

This implementation is based on Style-Based GAN in PyTorch, rosinality https://github.com/rosinality/style-based-gan-pytorch


## Requirments

h5py>=2.10.0
imgaug>=0.4.0
jupyterlab>=2.1.2
jupyterlab-server>=1.1.4
matplotlib>=3.2.1
notebook>=6.0.3
numpy>=1.18.4
opencv-python>=4.4.0.42
torch>=1.5.1+cu101
torchvision>=0.6.1+cu101

Usage:

1. Prepare MVTAD dataset

[Link](https://www.mvtec.com/company/research/datasets/mvtec-ad)

2. Dataset preprocessing

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER DATASET_PATH

3. Train generator

Run styleGAN.ipynb

4. Train anomaly mask geneartor

Run encoder.ipynb

5. Evaluate

Run evaluate.ipynb


Category | StyleGAN Generator model & Optimizer 
-----------|-------------------
Bottle      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Hazelnut      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Capsule      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Metal Nut      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Leather      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Pill      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Wood      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Carpet      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Tile      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Grid      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Cable      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Transistor      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Toothbrush      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Screw      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Zipper      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)

StyleGAN Generator & Optimizer checkpoints saved at resolution level 7 (512px * 512px)

Category | Anomaly Mask model & Optimizer 
-----------|-------------------
Bottle      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Hazelnut      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Capsule      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Metal Nut      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Leather      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Pill      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Wood      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Carpet      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Tile      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Grid      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Cable      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Transistor      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Toothbrush      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Screw      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)
Zipper      | [Link](https://drive.google.com/open?id=1QlXFPIOFzsJyjZ1AtfpnVhqW4Z0r8GLZ)

Model & Optimizer checkpoints saved at best performance.

## Sample

![Sample of the model trained on FFHQ](doc/sample_ffhq_new.png)
![Style mixing sample of the model trained on FFHQ](doc/sample_mixing_ffhq_new.png)

512px sample from the generator trained on FFHQ.

## Old Checkpoints

Resolution | Model & Optimizer | Running average of generator
-----------|-------------------|------------------------------
128px      | [Link](https://drive.google.com/open?id=1Fc0d8tTjS7Fcmr8gyHk8M0P-VMiRNeMl) | 100k iter [Link](https://drive.google.com/open?id=1b4MKSVTbWoY15NkzsM58T0QCvTE9d_Ch)
256px      | [Link](https://drive.google.com/open?id=1K2G1p-m1BQNoTEKJDBGAtFI1fC4eBjcd) | 140k iter [Link](https://drive.google.com/open?id=1n01mlc1mPpQyeUnnWNGeZiY7vp6JgakM)
512px      | [Link](https://drive.google.com/open?id=1Ls8NA56UnJWGJkRXXyJoDdz4a7uizBtw) | 180k iter [Link](https://drive.google.com/open?id=15lnKHnldIidQnXAlQ8PHo2W4XUTaIfq-)

Old version of checkpoints. As gradient penalty and discriminator activations are different, it is better to use new checkpoints to do some training. But you can use these checkpoints to make samples as generator architecture is not changed.

Running average of generator is saved at the specified iterations. So these two are saved at different iterations. (Yes, this is my mistake.)