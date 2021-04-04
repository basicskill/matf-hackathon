import numpy as np
from torch import nn
from torch.functional import F
from DAE.model import DAE

class MainModel(nn.Module):

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
