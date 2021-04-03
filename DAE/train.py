import torch
from torch import nn,optim
from torch.autograd import Variable as V
from torch.functional import F
import pandas as pd
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

if __name__ == "__main__":
    epochs = 10
    batch_size = 64

    input_size = 20
    latent_size = 8
    model = DAE(input_size, latent_size)

    data = pd.read_csv("model/training_test_data.csv")
    data.sample(frac = 1, random_state = 200)
    data = data.to_numpy()
    size = data.shape[0]


    training_data = data[: int(0.7*size)]
    validation_data = data[int(0.7*size) : int(0.9*size)]
    test_data = data[int(0.9*size) :]

    print(test_data)

    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     # train_loop()