# Variational Autoencoder

## What this is about
Just a simple implementation based on the VAE which seems to be an important thing for generative networks 

## What has been done 
1. Set up the Architecture
1. Set up loss function

## What else needs to be done
1. Set up the dataset and dataloader (mainly imagenet side)
1. Set up the training, which could be better implemented admittedly.
1. Set up validation, but only takes accuracy and loss. 
1. Results visualisation
1. Remove hardcoded weights

## How to run 

Make sure you change the directory of your data. I used the FashionMNIST and ImageNet. 

```
python -m src.main
```