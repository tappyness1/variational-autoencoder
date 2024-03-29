import torch
import numpy as np
from src.model import VAE

def recon_loss(logscale, decoded, dataset, x):
    if dataset == "Flowers102":
        # need to use MSE loss here
        # taken from here https://github.com/williamFalcon/pytorch-lightning-vae/blob/main/vae.py
        scale = torch.exp(logscale)
        dist = torch.distributions.Normal(decoded, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return -log_pxz.sum(dim=(1, 2, 3))
    
        # # commented out, but MSELoss can be computed here. not sure why we overcomplicate things
        # # https://ai.stackexchange.com/questions/27341/in-variational-autoencoders-why-do-people-use-mse-for-the-loss
    
        # mse = torch.nn.MSELoss(reduction='sum')
        # mse_loss = mse(decoded, x)
        # return - mse_loss

    if dataset == "FashionMNIST":
        # should be using MSE Loss here because FashionMNIST is a grayscale image
        # bce_loss = torch.nn.BCELoss(reduction='sum')

        # return bce_loss(decoded, x)

        mse = torch.nn.MSELoss(reduction='sum')
        return mse(decoded, x)

def kl_divergence(mean, log_var, z):
    # KL divergence between the latent distribution and the prior
    # here we are using the Monte Carlo KL divergence
    # not really sure what that means. 
    # I should really brush up on my prob and stats again sigh.
    std = torch.exp(0.5*log_var)    
    p = torch.distributions.Normal(torch.zeros_like(mean), torch.ones_like(std))
    q = torch.distributions.Normal(mean, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)

    # the alternative method is to do it like they do in the paper.
    # kl = 1/2 * (1 + log(std**2) - mean**2 - std((2)).sum(1)
    return kl

def elbo_loss(mean, log_var, z, log_scale, decoded, dataset, x):
    r_loss = recon_loss(log_scale, decoded, dataset, x)
    kl_loss = kl_divergence(mean, log_var, z)

    return (kl_loss + r_loss).mean()

if __name__ == "__main__":
    from src.model import VAE
    from src.dataset import get_load_data

    cfg_obj = {"dataset": {"dataset":"FashionMNIST"}}
    dataset = cfg_obj['dataset']['dataset']
    vae_model = VAE(cfg_obj)
    
    train, test = get_load_data(root = "../data", dataset = dataset, download = False)
      
    img, label = train[0] 
    

    img = img.unsqueeze(0)
    mean, log_var, z, decoded = vae_model(img)
    loss = elbo_loss(mean, log_var, z, vae_model.log_scale, decoded, dataset, img)
    loss.backward()
    print (loss)