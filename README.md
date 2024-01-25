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

## Resources

### From scratch implementation
https://www.youtube.com/watch?v=VELQT1-hILo - not right because he confused the MSE Loss and BCE Loss. See comment in video for more info. One good thing out of this is that he introduced me to the BCELoss https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html which has the 'reduction' arg.
https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py - good resource for imagenet dataset (ie 3xHxW)


### Actual Paper

https://arxiv.org/pdf/1312.6114.pdf - the start of the nightmare

### Helpful Understanding of the Calculations and motivation
https://www.youtube.com/watch?v=iwEzwTTalbg - good to download the PDF and follow along

### Loss Function
https://stats.stackexchange.com/questions/288451/why-is-mean-squared-error-the-cross-entropy-between-the-empirical-distribution-a/288453 - how gaussian likelihood becomes a mean squared error
https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss - more MSE issues
