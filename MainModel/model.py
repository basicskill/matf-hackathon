import numpy as np
from torch import nn
import torch
from torch.functional import F
from dae_model import DAE

class MainModel(nn.Module):

     def __init__(self, weather_encoder, pollution_encoder):
         super().__init__()
         self.weather_encoder = weather_encoder
         self.pollution_encoder = pollution_encoder

         self.layer1 = nn.Linear(18, 16)
         self.layer2 = nn.Linear(16, 8)
         self.layer3 = nn.Linear(8, 1)

     def forward(self, x):
         w1 = self.weather_encoder(x[:, 2:-7], encode=True)
         w2 = self.pollution_encoder(x[:, -7:], encode=True)
         feature = torch.cat([w1, w2, x[:, 0:2]], dim=1)
         h1 = F.relu(self.layer1(feature))
         h2 = F.relu(self.layer2(h1))

         return self.layer3(h2)
