import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
import numpy as np
from numba import jit

def addNoise(data):
    print(data.shape)
    mask = np.random.rand(data.shape[0], data.shape[1]) > 0.8
    data[mask] = -1
    return data


class AEDataset(Dataset):
    def __init__(self, data, transform = None, target_transform = None):
        # Input and target are same
        data_copy = np.copy(data)
        data_copy = addNoise(data_copy)
        self.X = [torch.tensor(row).float().to('cuda') for row in data_copy]
        self.y = [torch.tensor(row).float().to('cuda') for row in data]

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