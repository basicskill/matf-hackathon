import numpy as np
from torch import nn
from torch.functional import F

class DAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder_layer1 = nn.Linear(input_dim, 10)
        self.encoder_layer2 = nn.Linear(10, latent_dim)
        self.decoder_layer1 = nn.Linear(latent_dim, 10)
        self.decoder_layer2 = nn.Linear(10, input_dim)
    
    def encoder(self, x):
        h1 = F.relu(self.encoder_layer1(x))
        return F.relu(self.encoder_layer2(h1))

    def decoder(self, latent):
        h1 = F.relu(self.decoder_layer1(latent))
        return self.decoder_layer2(h1)

    def forward(self, x, encode):
        latent = self.encoder(x)

        # Run just encoding and prevent gradient from moving in this direction
        if encode:
            return np.c_[latent.detach().numpy()]
        
        # else, run decoder and allow propagation for training
        return self.decoder(latent)


class FCPolutionPredictor(nn.Module):

    def __init__(self, enc_input_dim, enc_latent_dim, fc_nodes, init_matrix):
        self.encoder = DAE(enc_input_dim, enc_latent_dim)

        self.fc = []
        layer_out_dim = enc_latent_dim

        for num_of_nodes in fc_nodes:
            self.fc.append(nn.Linear(layer_out_dim, num_of_nodes))
            layer_out_dim = num_of_nodes

    def forward(self, x):

        y = self.encoder(x, encode = True)

        for idx in range(len(self.fc)):
            y = self.fc[idx](y)
        
        return y