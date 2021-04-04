import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
import numpy as np
from numba import jit

class MainDataset(Dataset):
     def __init__(self, in_data, target_data):
         self.X = [torch.tensor(row) for row in in_data]
         self.y = [torch.tensor(row) for row in target_data]

     def __len__(self):
         assert(len(self.X) == len(self.y))
         return len(self.X)

     def __getitem__(self, idx):
         inp = self.X[idx]
         target = self.y[idx]

         sample = {"input": inp, "target": target}
         return sample