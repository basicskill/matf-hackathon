import torch
from torch.utils.data import DataLoader
from torch import nn,optim
from torch.autograd import Variable as V
from torch.functional import F
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from model import MainModel
from data_loaders import MDataset
from dae_model import DAE
from matplotlib import pyplot as plt

def train_loop(data_loader, wenc, penc, model, loss_fn, optimizer):
    model.train()

    size = len(data_loader.dataset)
    for batch, (X, y) in enumerate(data_loader):
        tmp = wenc(X[:, 2:36], encode=True)
        pred = model(X)
        loss = loss_fn(pred, y)

        # reset gradients to avoid accumulation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def valid_loop(data_loader, wenc, penc, model, loss_fn):
    model.eval()
    size = len(data_loader.dataset)
    valid_loss = 0

    losses = []
    with torch.no_grad():
        for batch, (X, y) in enumerate(data_loader):
            pred = model(X)
            loss = loss_fn(pred, y).item()
            losses.append(loss)

    print(f"Validation avg loss: {sum(losses) / len(losses):>8f} \n")
    with torch.no_grad():
        pred = model(X)
        print(y[0])
        print(pred[0])
    return sum(losses) / len(losses)

def test(data_loader, wenc, penc, model, loss_fn):
    model.eval()
    size = len(data_loader.dataset)
    test_loss = 0

    preds = []
    targets = []
    with torch.no_grad():
        for X, y in data_loader:
            pred = model(X)
            preds.append(pred[0])
            targets.append(y[0])
            test_loss += loss_fn(pred, y).item()

    test_loss /= size
    print(f"Test avg loss: {test_loss:>8f} \n")

    preds = np.array(preds)
    np.savetxt("MainModel/preds.txt", preds)
    targets = np.array(targets)
    np.savetxt("MainModel/targets.txt", targets)
    plt.plot(preds, label = 'predictions')
    plt.plot(targets, label = 'GT')
    plt.legend()
    plt.show()


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
    neg_mask = np.isnan(in_data)
    for col in range(in_data.shape[1]):
        max_mask = in_data[:, col] > max_values[col]
        min_mask = in_data[:, col] < min_values[col]
        
        out_data[:, col] = in_data[:, col] - min_values[col] #/ (max_values[col] - min_values[col])
    
        if max_values[col] - min_values[col] != 0:
            out_data[:, col] /= max_values[col] - min_values[col]

        np.place(out_data[:, col], max_mask, 1.0)
        np.place(out_data[:, col], min_mask, 0.0)
    
    np.place(out_data, neg_mask, -1.0)
    return out_data

if __name__ == "__main__":
    epochs = 5000
    batch_size = 128

    input_size = 34
    latent_size = 8
    
    wenc = DAE(34, 8)
    penc = DAE(7, 8)
    wenc = torch.load("MainModel/wenc.pth")
    penc = torch.load("MainModel/penc.pth")

    # model = MainModel(wenc, penc)
    model = torch.load("MainModel/train/model2659.pth")
    model.to('cuda')
    torch.backends.cudnn.benchmark = True
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    DECAY = 0.95
    scheduler = LambdaLR(optimizer, lr_lambda = lambda t : DECAY**t)

    data = pd.read_csv("model/FCData.csv")
    target = pd.read_csv("model/FCTargets.csv")
    
    data = data.to_numpy()
    data[:, 0] = data[:, -1]
    data = np.delete(data, -1, 1)
    target = target.to_numpy()
    target = target[:, 1:2]

    all_data = np.concatenate([data, target], axis=1)
    mask = np.isnan(all_data[:, -1])
    print(mask.shape)
    mask = np.invert(mask)
    print(np.sum(mask))
    print(all_data.shape)
    all_data = all_data[mask]
    print(all_data.shape)
    size = data.shape[0]

    training_data = all_data[: int(0.7*size)]
    validation_data = all_data[int(0.7*size) : int(0.9*size)]
    test_data = all_data[int(0.9*size) :]

    # Train data clean
    # training_data = training_data[~np.isnan(training_data).any(axis=1)]
    # np.argwhere(np.isnan(training_data))
    min_values, max_values = find_min_max(training_data)
    # override month and hour max values
    min_values[0] = 1
    min_values[1] = 0
    max_values[0] = 12
    max_values[1] = 23
    training_data = apply_min_max(training_data, min_values, max_values)

    # Validation data
    # validation_data = validation_data[~np.isnan(validation_data).any(axis=1)]
    # np.argwhere(np.isnan(validation_data))
    validation_data = apply_min_max(validation_data, min_values, max_values)

    # # Test data
    # test_data = test_data[~np.isnan(test_data).any(axis=1)]
    # np.argwhere(np.isnan(test_data))
    test_data = apply_min_max(test_data, min_values, max_values)

    training_dataset = MDataset(training_data)
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)

    X, y = next(iter(training_loader))
    print(X[0])
    print(y[0])

    validation_dataset = MDataset(validation_data)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    test_dataset = MDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # valid_losses = []
    # for t in range(epochs):
    #     print(f"Epoch {t+1}\n-------------------------------")
    #     train_loop(training_loader, wenc, penc, model, loss_fn, optimizer)
    #     val_loss = valid_loop(validation_loader, wenc, penc, model, loss_fn)
    #     print(optimizer.param_groups[0]['lr'])
    #     torch.save(model, "MainModel/train/model" + str(t) + ".pth")
    #     valid_losses.append(val_loss)

    #     if t % 200 == 0:    
    #         scheduler.step()
    
    # np.savetxt("MainModel/train/valid.txt", np.array(valid_losses))
    test(test_loader, wenc, penc, model, loss_fn)