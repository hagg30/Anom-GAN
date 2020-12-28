# Anomaly Mask GAN

Implementation of AnoM-GAN: A Novel Unsupervised Anomaly Localization with GAN Manifold Mask Model in PyTorch

This implementation is based on Style-Based GAN in PyTorch, rosinality https://github.com/rosinality/style-based-gan-pytorch


## Requirments

Please check requirements.txt

## Usage

1. Prepare MVTAD dataset

[Link](https://www.mvtec.com/company/research/datasets/mvtec-ad)

2. Dataset preprocessing

> python prepare_data.py --out LMDB_PATH --n_worker N_WORKER DATASET_PATH

Place checkpoints on checkpt folder.

3. Evaluate

Run evaluate.ipynb

Modify target data label

Modify checkpoint/test data location


5. Train your own generator

Run styleGAN.ipynb

6. Train your own anomaly mask geneartor

Run encoder.ipynb

## Pretrained Checkpoints

StyleGAN Generator & Optimizer checkpoints saved at resolution level 7 (512px * 512px)

[Link](https://www.dropbox.com/sh/f6i0w7tyvhw969v/AABb6rPslJ-2aurn6aa_7YNVa?dl=0)

Corressponding Mask Generator & Encoder and Optimizer checkpoints saved at best performance.

[Link](https://www.dropbox.com/sh/vkori9qll8uwszn/AAA6GrmDIXPdZw7YZEIHL-PKa?dl=0)


StyleGAN Generator Samples

[Link](https://www.dropbox.com/sh/nsnaib0xl5gkd5h/AACMCvJBcudXAqvm5fZsPSKga?dl=0)
