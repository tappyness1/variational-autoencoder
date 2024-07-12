from torch.nn import Conv2d, ReLU, MaxPool2d, ConvTranspose2d, BatchNorm2d, AdaptiveAvgPool2d, Linear, Softmax
import torch.nn as nn
import torch
import numpy as np

def reparameterise(mean, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    return mean + eps*std

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
            self.encoder_decoder = EncoderDecoderFMNIST()

        if cfg_obj["dataset"]['dataset'] == "Flowers102":
            # in the case of Flowers102, which are 3x224x224 images, we do conv layers then flatten
            self.encoder_decoder = EncoderDecoderFlowers102()

        # some papers say that your sigma can be a trainable parameter
        # some say you can just set it to 1 or any number you want. 
        self.log_scale = nn.Parameter(torch.Tensor([1.0]))

    def forward(self, input):

        return self.encoder_decoder(input)
    
class EncoderDecoderFMNIST(nn.Module):
    def __init__(self):
        super(EncoderDecoderFMNIST, self).__init__()
        relu = nn.ReLU()
        encoder_params = [28*28, 64, 32, 16, 8, 4, 2]
        self.encoder = nn.Sequential()
        for i in range(len(encoder_params)-1):
            self.encoder.add_module(f"encoder_{i}", nn.Linear(encoder_params[i], encoder_params[i+1]))
            self.encoder.add_module(f"encoder_relu_{i}", relu)

        self.decoder = nn.Sequential()
        for i in range(len(encoder_params)-1, 0, -1):
            self.decoder.add_module(f"decoder_{i}", nn.Linear(encoder_params[i], encoder_params[i-1]))
            self.decoder.add_module(f"decoder_relu_{i}", relu)

        self.FC_mean = nn.Linear(2 , 2)
        self.FC_logvar = nn.Linear(2 , 2)

    def forward(self, input):
        input = input.reshape(-1, 28*28)
        encoded = self.encoder(input)
        mean = self.FC_mean(encoded)
        log_var = self.FC_logvar(encoded)

        z = reparameterise(mean, log_var)

        decoded = self.decoder(z)
        decoded = decoded.reshape(-1, 1, 28, 28)
        return mean, log_var, z, decoded
    
class EncoderDecoderFlowers102(nn.Module):
    def __init__(self) -> None:
        super(EncoderDecoderFlowers102, self).__init__()
        relu = nn.ReLU()

        self.encoder = nn.Sequential()
        encoder_conv_params = [3,64,32,16,8,4,2]
        encoder_lin_params = [89888, 1024, 20]

        for i in range(len(encoder_conv_params)-1):
            self.encoder.add_module(f"encoder_{i}", nn.Conv2d(encoder_conv_params[i], encoder_conv_params[i+1], 3))
            self.encoder.add_module(f"encoder_relu_{i}", relu)

        self.encoder.add_module("flatten", nn.Flatten())

        for i in range(len(encoder_lin_params)-1):
            self.encoder.add_module(f"encoder_lin_{i}", nn.Linear(encoder_lin_params[i], encoder_lin_params[i+1]))
            self.encoder.add_module(f"encoder_lin_relu_{i}", relu)

        self.FC_mean = nn.Linear(20 , 20)
        self.FC_logvar = nn.Linear(20 , 20)

        self.decoder_linear = nn.Sequential()
        for i in range(len(encoder_lin_params)-1, 0, -1):
            self.decoder_linear.add_module(f"decoder_lin_{i}", nn.Linear(encoder_lin_params[i], encoder_lin_params[i-1]))
            self.decoder_linear.add_module(f"decoder_lin_relu_{i}", relu)
        
        self.decoder_conv = nn.Sequential()
        for i in range(len(encoder_conv_params)-1, 0, -1):
            self.decoder_conv.add_module(f"decoder_{i}", nn.ConvTranspose2d(encoder_conv_params[i], encoder_conv_params[i-1], 3))
            self.decoder_conv.add_module(f"decoder_relu_{i}", relu)

    def forward(self, input):        
            
        encoded = self.encoder(input)
        mean = self.FC_mean(encoded)
        log_var = self.FC_logvar(encoded)
        z = reparameterise(mean, log_var)

        decoded = self.decoder_linear(z)
        decoded = decoded.reshape(-1, 2, 212, 212)
        decoded = self.decoder_conv(decoded)
        
        return mean, log_var, z, decoded

if __name__ == "__main__":
    import numpy as np
    import torch
    from torchsummary import summary

    np.random.seed(42)
    torch.manual_seed(42)    

    X = np.random.rand(5, 3, 224, 224).astype('float32')
    
    X = torch.tensor(X)

    cfg_obj = {"dataset": {"dataset":"Flowers102"}}
    model = VAE(cfg_obj = cfg_obj)
    
    # summary(model, (3, 224, 224))
    print ()
    print (model.forward(X)[0].shape)