import multiprocessing
import torch
from torch.utils import data
from functools import partial
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pickle
import numpy as np

import os
PATH = os.getcwd()

class DIDataset(torch.utils.data.Dataset):

    def __init__(self, X_Train, Y_Train, transform=None):
        self.X_Train = X_Train
        self.Y_Train = Y_Train
        self.transform = transform

    def __len__(self):
        return len(self.X_Train)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.X_Train[idx]
        y = self.Y_Train[idx]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y
    
def double_integrator_datasets(num_val = 400, num_test = 200, dataset_name = "default"):
    with open(PATH + "/nfl_robustness_training/src/_static/datasets/double_integrator/" + dataset_name + "/dataset.pkl", 'rb') as f:
        xs, us = pickle.load(f)
    
    # with open(PATH + "/nfl_robustness_training/src/_static/datasets/double_integrator/" + dataset_name + "/us.pkl", 'rb') as f:
    #     us = pickle.load(f)

    num_train = xs.shape[0] - num_val - num_test

    dataset = DIDataset(torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32), transform=None)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])
    return train_set, val_set, test_set

def double_integrator_loaders(
        batch_size, 
        shuffle_train = True, 
        shuffle_test = False, 
        normalize_input = False, 
        num_examples = None, 
        test_batch_size=None, 
        num_val = 400, 
        num_test = 200,
        dataset_name = 'default'
        ): 
    # file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator_train/xs.pkl', 'rb')
    # X_train = pickle.load(file)
    # file.close()

    # file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator_train/us.pkl', 'rb')
    # Y_train = pickle.load(file)
    # file.close()

    # train_dataset = DIDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(Y_train, dtype=torch.float32), transform=None)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, num_workers=0)


    # file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator_val/xs.pkl', 'rb')
    # X_test = pickle.load(file)
    # file.close()

    # file = open('/home/nick/Documents/code/nfl_veripy/nfl_veripy/src/nfl_veripy/_static/datasets/double_integrator_val/us.pkl', 'rb')
    # Y_test = pickle.load(file)
    # file.close()

    # test_dataset = DIDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(Y_test, dtype=torch.float32), transform=None)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle_test, num_workers=0)

    # if test_batch_size:
    #     batch_size = test_batch_size

    # std = [1.0]
    # mean = [0.0]
    # train_loader.std = std
    # test_loader.std = std
    # train_loader.mean = mean
    # test_loader.mean = mean

    train_set, val_set, test_set = double_integrator_datasets(num_val, num_test, dataset_name)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle_test, num_workers=0)


    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    with open(PATH + '/nfl_robustness_training/_static/datasets/double_integrator/xs.pkl', 'rb') as f:
        xs = pickle.load(f)
    
    with open(PATH + '/nfl_robustness_training/_static/datasets/double_integrator/us.pkl', 'rb') as f:
        us = pickle.load(f)

    num_test = 200
    num_val = 400
    num_train = xs.shape[0] - num_val - num_test

    dataset = DIDataset(torch.tensor(xs, dtype=torch.float32), torch.tensor(us, dtype=torch.float32), transform=None)
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

    xs_train, us_train = train_set[:]
    xs_val, us_val = val_set[:]
    xs_test, us_test = test_set[:]
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(xs_train[:, 0], xs_train[:, 1], us_train[:, 0], c = 'b', marker='o')
    ax.scatter(xs_val[:, 0], xs_val[:, 1], us_val[:, 0], c = 'r', marker='o', zorder=10)
    ax.scatter(xs_test[:, 0], xs_test[:, 1], us_test[:, 0], c = 'g', marker='o')
    plt.show()

    # import pdb; pdb.set_trace()