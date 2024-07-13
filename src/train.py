
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

from src.dataset import get_load_data
from src.loss_function import elbo_loss
from src.model import VAE


def train(train_set, cfg):
           
    model = VAE(cfg_obj = cfg)

    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(device)

    if cfg['show_model_summary']:
        summary(model, (3,224,224))

    train_dataloader = DataLoader(train_set, batch_size=cfg['train']['batch_size'], shuffle = True)
    dataset = cfg['dataset']['dataset']
    
    for epoch in range(cfg['train']['epochs']):
        print (f"Epoch {epoch + 1}:")
        # for i in tqdm(range(X.shape[0])):
        overall_loss = []
        with tqdm(train_dataloader) as tepoch:
            for imgs, classes in tepoch:
                imgs = imgs.to(device)
                
                mean, log_var, z, decoded = model(imgs)
                loss = elbo_loss(mean, log_var, decoded, imgs)
                
                optimizer.zero_grad() 
                loss.backward()
                optimizer.step()
                overall_loss.append(loss.item())
                tepoch.set_postfix(loss=loss.item())
                
            print(f"Epoch {epoch + 1} loss: {sum(overall_loss) / (len(overall_loss)*cfg['train']['batch_size'])}")    

    print("training done")
    torch.save(model, cfg['save_model_path'])

    return model

if __name__ == "__main__":

    torch.manual_seed(42)

    cfg = {"save_model_path": "model_weights/vae_mnist.pt",
           'show_model_summary': False, 
           'train': {"epochs": 50, 'lr': 0.001, 'weight_decay': 5e-3, 
                    'batch_size': 4096},
           'dataset': {"dataset": "MNIST"}}

    # cfg = {"save_model_path": "model_weights/vae_flowers.pt",
    #        'show_model_summary': False, 
    #        'train': {"epochs": 3, 'lr': 0.005, 'weight_decay': 5e-3},
    #        'dataset': {"dataset": "Flowers102"}}  

    # train_set, _ = get_load_data(root = "../data", dataset = cfg['dataset']['dataset'])
    train_set, _ = get_load_data(root = "/content/data", dataset = cfg['dataset']['dataset'], download = True)
    train(train_set = train_set, cfg = cfg)

    # cannot use FashionMNIST because size needs to be 224x224x3 at the very least
    # train_set, test_set = get_load_data(root = "../data", dataset = "FashionMNIST")
    # train(epochs = 1, train_set = train_set, in_channels = 1, num_classes = 10)
    