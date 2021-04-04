import numpy as np
from torch import nn
from torch.functional import F

class DAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder_layer1 = nn.Linear(input_dim, 32)
        self.encoder_layer2 = nn.Linear(32, 16)
        self.encoder_layer3 = nn.Linear(16, latent_dim)

        self.decoder_layer1 = nn.Linear(latent_dim, 16)
        self.decoder_layer2 = nn.Linear(16, 32)
        self.decoder_layer3 = nn.Linear(32, input_dim)
    
    def encoder(self, x):
        h1 = F.relu(self.encoder_layer1(x))
        h2 = F.relu(self.encoder_layer2(h1))
        return F.relu(self.encoder_layer3(h2))

    def decoder(self, latent):
        h1 = F.relu(self.decoder_layer1(latent))
        h2 = F.relu(self.decoder_layer2(h1))
        return self.decoder_layer3(h2)

    def forward(self, x, encode):
        latent = self.encoder(x)
        # Run just encoding and prevent gradient from moving in this direction
        if encode:
            return latent.detach()
        
        # else, run decoder and allow propagation for training
        return self.decoder(latent)