from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
import numpy as np
    
class EncoderDecoder(nn.Module):
    def __init__(self, cfg_obj):
        super(EncoderDecoder, self).__init__()
        self.cfg_obj = cfg_obj
        if cfg_obj["dataset"]['dataset'] == "FashionMNIST":

            # in the case of FashionMNIST, which are 28x28 images, we do linear layers
            self.encoder = nn.Sequential(
                nn.Linear(28*28, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2),
                nn.ReLU(),
            )

            self.decoder = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 28*28),
                nn.ReLU(),
            )

        if cfg_obj["dataset"]['dataset'] == "Flowers102":

            # in the case of Flowers102, which are 3x224x224 images, we do conv layers then flatten

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3), # 222
                nn.ReLU(),
                nn.Conv2d(64, 32, 3), # 220
                nn.ReLU(),
                nn.Conv2d(32, 16, 3), # 218
                nn.ReLU(),
                nn.Conv2d(16, 8, 3), # 216
                nn.ReLU(),
                nn.Conv2d(8, 4, 3), # 214
                nn.ReLU(),
                nn.Conv2d(4, 2, 3), # 212
                nn.ReLU(),
                nn.Flatten(), # 2*212*212 = 89888
                nn.Linear(89888, 1024),
                nn.ReLU(),
                nn.Linear(1024, 256),
                nn.ReLU()
            )
            self.decoder_linear = nn.Sequential(
                nn.Linear(256, 1024),
                nn.ReLU(),
                nn.Linear(1024, 89888),
                nn.ReLU()
            )
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(2, 4, 3), # 214
                nn.ReLU(),
                nn.ConvTranspose2d(4, 8, 3), # 216
                nn.ReLU(),
                nn.ConvTranspose2d(8, 16, 3), # 218
                nn.ReLU(),  
                nn.ConvTranspose2d(16, 32, 3), # 220
                nn.ReLU(),
                nn.ConvTranspose2d(32, 64, 3), # 222
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 3), # 224
            )

    def forward(self, input):
        
        encoded = self.encoder(input)
        
        if self.cfg_obj["dataset"]['dataset'] == "FashionMNIST": 
            decoded = self.decoder(encoded)

        if self.cfg_obj["dataset"]['dataset'] == "Flowers102":
            decoded = self.decoder_linear(encoded)
            decoded = decoded.reshape(-1, 2, 212, 212)
            decoded = self.decoder_conv(decoded)
        
        return encoded, decoded

class Model(nn.Module):
    def __init__(self, cfg_obj):
        self.encoder_decoder = EncoderDecoder()

    def forward(self, input):
        encoded, decoded = self.encoder_decoder(input)
        
        return encoded, decoded

class VAE(nn.Module):
    def __init__(self, cfg_obj):
        super(VAE, self).__init__()
        self.cfg_obj = cfg_obj

        # get the encoder and decoder
        # output of the encoder will then go through another 2 separate FC
        # first is to get mu, the mean
        # second is to get the variance
        # for numerical stability, we will obtain the log variance instead
        # read more here - https://stats.stackexchange.com/questions/353220/why-in-variational-auto-encoder-gaussian-variational-family-we-model-log-sig

        if cfg_obj["dataset"]['dataset'] == "FashionMNIST":

            # in the case of FashionMNIST, which are 28x28 images, we do linear layers
            self.encoder = nn.Sequential(
                nn.Linear(28*28, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 4),
                nn.ReLU(),
                nn.Linear(4, 2)
            )
            self.FC_mean = nn.Linear(2 , 2)
            self.FC_logvar = nn.Linear(2 , 2)

            self.decoder = nn.Sequential(
                nn.Linear(2, 4),
                nn.ReLU(),
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 28*28),
                nn.ReLU(),
            )

        if cfg_obj["dataset"]['dataset'] == "Flowers102":

            # in the case of Flowers102, which are 3x224x224 images, we do conv layers then flatten

            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3), # 222
                nn.ReLU(),
                nn.Conv2d(64, 32, 3), # 220
                nn.ReLU(),
                nn.Conv2d(32, 16, 3), # 218
                nn.ReLU(),
                nn.Conv2d(16, 8, 3), # 216
                nn.ReLU(),
                nn.Conv2d(8, 4, 3), # 214
                nn.ReLU(),
                nn.Conv2d(4, 2, 3), # 212
                nn.ReLU(),
                nn.Flatten(), # 2*212*212 = 89888
                nn.Linear(89888, 1024),
                nn.ReLU(),
                nn.Linear(1024, 20),
            )
            self.FC_mean = nn.Linear(20 , 20)
            self.FC_logvar = nn.Linear(20 , 20)

            self.decoder_linear = nn.Sequential(
                nn.Linear(20, 1024),
                nn.ReLU(),
                nn.Linear(1024, 89888),
                nn.ReLU()
            )
            self.decoder_conv = nn.Sequential(
                nn.ConvTranspose2d(2, 4, 3), # 214
                nn.ReLU(),
                nn.ConvTranspose2d(4, 8, 3), # 216
                nn.ReLU(),
                nn.ConvTranspose2d(8, 16, 3), # 218
                nn.ReLU(),  
                nn.ConvTranspose2d(16, 32, 3), # 220
                nn.ReLU(),
                nn.ConvTranspose2d(32, 64, 3), # 222
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 3), # 224
            )
            # some papers say that your sigma can be a trainable parameter
            # some say you can just set it to 1 or any number you want. 
            # I'm a sheep so here we go.
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def reparameterise(self, mean, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mean + eps*std

    def forward(self, input):
        if self.cfg_obj["dataset"]['dataset'] == "FashionMNIST": 
            input = input.reshape(-1, 28*28)
        
        encoded = self.encoder(input)
        mean = self.FC_mean(encoded)
        log_var = self.FC_logvar(encoded)
        z = self.reparameterise(mean, log_var)
        
        if self.cfg_obj["dataset"]['dataset'] == "FashionMNIST": 
            decoded = self.decoder(z)
            decoded = decoded.reshape(-1, 1, 28, 28)

        if self.cfg_obj["dataset"]['dataset'] == "Flowers102":
            decoded = self.decoder_linear(z)
            decoded = decoded.reshape(-1, 2, 212, 212)
            decoded = self.decoder_conv(decoded)
        
        return mean, log_var, z, decoded

# class VAE(nn.Module):
#     def __init__(self, cfg_obj):
#         super(VAE, self).__init__()
#         self.encoder_decoder = VAE_util(cfg_obj)

#     def forward(self, input):
#         mean, log_var, z, decoded = self.encoder_decoder(input)
        
#         return mean, log_var, z, decoded


if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')
    
    X = torch.tensor(X)

    cfg_obj = {"dataset": {"dataset":"Flowers102"}}
    model = EncoderDecoder(cfg_obj = cfg_obj)
    
    summary(model, (3, 224, 224))
    print ()
    print (model.forward(X)[0].shape)