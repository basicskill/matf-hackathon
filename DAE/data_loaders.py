import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd
import numpy as np
from numba import jit

def addNoise(data):
    data_copy = data.clone()
    mask = torch.cuda.FloatTensor(data.shape).uniform_() > 0.2
    data_copy[mask] = -1
    return data_copy


class AEDataset(Dataset):
    def __init__(self, data, transform = None, target_transform = None):
        # Input and target are same
        self.X = [torch.tensor(row).float().to('cuda') for row in data]
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

        sample = {"input": inp, "target": target}
        return sample

class FCDataset(Dataset):
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