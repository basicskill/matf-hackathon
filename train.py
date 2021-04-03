import torch
from torch import nn,optim
from torch.autograd import Variable as V
from torch.functional import F
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from model import DAE

def train_loop(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        pred = model(X, encode = False)
        loss = loss_fn(pred, y)

        # reset gradients to avoid accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def valid_loop(data_loader, model, loss_fn):
    size = len(data_loader.dataset)
    valid_loss = 0

    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X, encode = False)
            valid_loss += loss_fn(pred, y).item()

    valid_loss /= size
    print(f"Validation avg loss: {valid_loss:>8f} \n")

def find_min_max(in_data):
    num_of_attr = in_data.shape[1]

    min_values = np.zeros(num_of_attr)
    max_values = np.zeros(num_of_attr)

    for col in range(num_of_attr):
        min_values[col] = np.nanmin(in_data[:, col])
        max_values[col] = np.nanmax(in_data[:, col])

    return min_values, max_values

def apply_min_max(in_data, min_values, max_values):

    out_data = in_data.copy()
    

    for col in range(in_data.shape[1]):
        
        out_data[:, col] = in_data[:, col] - min_values[col] #/ (max_values[col] - min_values[col])

        if max_values[col] - min_values[col] != 0:
            out_data[:, col] /= max_values[col] - min_values[col]

    return out_data

if __name__ == "__main__":
    epochs = 10
    batch_size = 64

    input_size = 20
    latent_size = 8
    model = DAE(input_size, latent_size)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # train_loop()