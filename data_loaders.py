import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
import pandas as pd

class DataLoader(Dataset):
    def __init__(self, data_file, transform = None, target_transform = None):
        data = pd.read_csv(data_file)

        # Input and target are same
        self.X = data
        self.y = data

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