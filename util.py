import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
# transforms
from torchvision import transforms as T

from torch import sigmoid
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.conv import TransformerConv
from torch.utils.data import DataLoader 
from matplotlib.image import imsave
import torch_geometric
import torchvision
from torchvision import datasets, transforms
from model import Model

def setup_model(device, model_path):
    torch.manual_seed(0)
    model = Net()
    model.load_state_dict(torch.load(model_path))
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def load_checkpoint(device, checkpoint_path):
    torch.manual_seed(0)
    checkpoint = torch.load(checkpoint_path)
    model = Model()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    epoch_loss = checkpoint['epoch_loss']
    return model, optimizer, epoch, epoch_loss

def load_model(device, model_path):
    torch.manual_seed(0)
    model = Model()
    model = torch.load(model_path)
    print("Loaded the parameters for the model from %s"%model_path)
    model.to(device)
    return model

def new_model(device):
    torch.manual_seed(0)
    model = Model()
    model.to(device)
    return model
def load_images(device):
    import scipy.io
    train_data = scipy.io.loadmat('/data/tian/MSTAR/mat/train128.mat')
    test_data = scipy.io.loadmat('/data/tian/MSTAR/mat/test128.mat')

    X_train, y_train = np.array(train_data['train_data']), np.array(train_data['train_label'])
    X_test, y_test =  np.array(test_data['test_data']), np.array(test_data['test_label'])
    
    # Train test split
    # X = np.concatenate((X_train, X_test), axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    
    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    X_train_image = X_train
    X_test_image = X_test

    X_train_image = (X_train_image - X_train_image.min(axis=(1,2)).reshape([-1,1,1]))/X_train_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_train_image), np.max(X_train_image))
    X_train_image = X_train_image[:, np.newaxis, :, :]
    X_train_image = torch.from_numpy(X_train_image).to(device)
    X_train_image = X_train_image.type(torch.float32)
    
    X_test_image = (X_test_image - X_test_image.min(axis=(1,2)).reshape([-1,1,1]))/X_test_image.ptp(axis=(1,2)).reshape([-1,1,1])
    print(np.min(X_test_image), np.max(X_test_image))
    X_test_image = X_test_image[:, np.newaxis, :, :]
    X_test_image = torch.from_numpy(X_test_image).to(device)
    X_test_image = X_test_image.type(torch.float32)
   
    if y_train.min().item == 1 and y_train.max().item() == 10:
        train_label = torch.from_numpy(y_train).to(device) - 1
    else:
        train_label = torch.from_numpy(y_train).to(device)
    train_label = train_label.type(torch.int64)
    
    if y_test.min().item == 1 and y_test.max().item() == 10:
        test_label = torch.from_numpy(y_test).to(device) - 1
    else:
        test_label = torch.from_numpy(y_test).to(device)
    test_label = test_label.type(torch.int64)

    print('X_train_size: ', X_train_image.size(), 'y_train_size: ', train_label.size())
    print('X_test_size: ', X_test_image.size(), 'y_test_size: ', test_label.size())
    
    train_data = [[X_train_image[i], train_label[i]] for i in range(train_label.size()[0])]
    test_data = [[X_test_image[i], test_label[i]] for i in range(test_label.size()[0])]

    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)

    return train_dataloader, test_dataloader

def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.MNIST(root='/data/jacob/MNIST/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='/data/jacob/MNIST/', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader
    
def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.FashionMNIST(root='/data/jacob/FashionMNIST/', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.FashionMNIST(root='/data/jacob/FashionMNIST/', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader