import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from torch.autograd import Variable as V
from torch.functional import F
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from model import DAE
from data_loaders import AEDataset, addNoise

def train_loop(data_loader, model, loss_fn, optimizer):
    model.train()

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
    model.eval()
    size = len(data_loader.dataset)
    valid_loss = 0

    losses = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            pred = model(X, encode = False)
            loss = loss_fn(pred, y).item()
            losses.append(loss)

    print(f"Validation avg loss: {sum(losses) / len(losses):>8f} \n")
    with torch.no_grad():
        (X, y) = next(iter(data_loader))
        pred = model(X, encode = False)
        print(y[0])
        print(pred[0])
    return sum(losses) / len(losses)

def test(data_loader, model, loss_fn):
    model.eval()
    size = len(data_loader.dataset)
    test_loss = 0

    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X, encode = False)
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test avg loss: {test_loss:>8f} \n")

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
        max_mask = in_data[:, col] > max_values[col]
        min_mask = in_data[:, col] < min_values[col]
        out_data[:, col] = in_data[:, col] - min_values[col] #/ (max_values[col] - min_values[col])
    
        if max_values[col] - min_values[col] != 0:
            out_data[:, col] /= max_values[col] - min_values[col]

        np.place(out_data[:, col], max_mask, 1.0)
        np.place(out_data[:, col], min_mask, 0.0)
    return out_data

if __name__ == "__main__":
    epochs = 5000
    batch_size = 128

    input_size = 34
    latent_size = 8
    
    model = DAE(input_size, latent_size)
    model.to('cuda')
    torch.backends.cudnn.benchmark = True
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    DECAY = 0.95
    scheduler = LambdaLR(optimizer, lr_lambda = lambda t : DECAY**t)


    data = pd.read_csv("model/training_test_data.csv")
    data.sample(frac = 1, random_state = 200)
    data = data.to_numpy()
    size = data.shape[0]

    training_data = data[: int(0.7*size)]
    validation_data = data[int(0.7*size) : int(0.9*size)]
    test_data = data[int(0.9*size) :]

    # Train data clean
    weather_train = training_data[:, 5:-7]
    weather_train = weather_train[~np.isnan(weather_train).any(axis=1)]
    np.argwhere(np.isnan(weather_train))
    min_values, max_values = find_min_max(weather_train)
    weather_train = apply_min_max(weather_train, min_values, max_values)

    # Validation data
    weather_valid = validation_data[:, 5:-7]
    weather_valid = weather_valid[~np.isnan(weather_valid).any(axis=1)]
    np.argwhere(np.isnan(weather_valid))
    weather_valid = apply_min_max(weather_valid, min_values, max_values)

    # Test data
    weather_test = test_data[:, 5:-7]
    weather_test = weather_test[~np.isnan(weather_test).any(axis=1)]
    np.argwhere(np.isnan(weather_test))
    weather_test = apply_min_max(weather_test, min_values, max_values)

    training_dataset = AEDataset(weather_train)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)

    validation_dataset = AEDataset(weather_valid)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)

    test_dataset = AEDataset(weather_test)
    test_loader = DataLoader(test_dataset)

    valid_losses = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(training_loader, model, loss_fn, optimizer)
        val_loss = valid_loop(validation_loader, model, loss_fn)
        print(optimizer.param_groups[0]['lr'])
        torch.save(model, "DAE/weather_train/model" + str(t) + ".pth")
        valid_losses.append(val_loss)

        if t % 200 == 0:    
            scheduler.step()
    
    np.savetxt("DAE/weather_train/valid.txt", np.array(valid_losses))
    test(test_loader, model, loss_fn)