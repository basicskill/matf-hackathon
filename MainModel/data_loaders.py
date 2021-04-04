import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
import numpy as np
from numba import jit

class MDataset(Dataset):
    def __init__(self, data, transform = None, target_transform = None):
        # Input and target are same
        self.X = [torch.tensor(row[:-1]).float().to('cuda') for row in data]
        self.y = [torch.tensor(row[-1:]).float().to('cuda') for row in data]

        # Keep target transform always None
        # Input transform should be non for basic AE, but noising function for DAE
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        assert(len(self.X) == len(self.y))
        return len(self.X)

    def __getitem__(self, idx):
        inp = self.X[idx]
        target = self.y[idx]

        if self.transform:
            inp = self.transform(inp)

        if self.target_transform:
            assert(False)

        sample = [inp, target]
        return sample