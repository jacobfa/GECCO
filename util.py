import numpy as np
import scipy.misc as im
import os
import torch
# data data loader
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms as transforms

def load_images(width=128, height=128, crop_size=128, aug=False):
    sub_dir = ["2S1", "BMP2", "BRDM2", "BTR60", "BTR70", "D7", "T62", "T72", "ZIL131", "ZSU_23_4"]
    X = []
    y = []
    
    data_dir = '/Users/jacobfa/Downloads/MSTAR-10/'
    train_dir = data_dir + "train/"
    test_dir = data_dir + "test/"
     
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    
    for i in range(len(sub_dir)):
        tmp_dir = train_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
        # print(sub_dir[i], len(img_idx))
        y_train += [i] * len(img_idx)
        for j in range(len(img_idx)):
            img = Image.open(tmp_dir + img_idx[j])
            # grayscale
            img = img.convert('L')
            img = img.resize((height, width))
            img = np.array(img)
            # img = img[16:112, 16:112]   # crop
            X_train.append(img)
                
    for i in range(len(sub_dir)):
        tmp_dir = test_dir + sub_dir[i] + "/"
        img_idx = [x for x in os.listdir(tmp_dir) if x.endswith(".jpeg")]
        # print(sub_dir[i], len(img_idx))
        y_test += [i] * len(img_idx)
        for j in range(len(img_idx)):
            img = Image.open(tmp_dir + img_idx[j])
            # grayscale
            img = img.convert('L')
            img = img.resize((height, width))
            img = np.array(img)
            # img = img[16:112, 16:112]   # crop
            X_test.append(img)
    

    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train ,dtype=int)
    X_test = np.array(X_test ,dtype=float)
    y_test = np.array(y_test ,dtype=int)

    # expand first dimension for channel
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    
    # to float32
    train_data = [(X_train[i].astype(np.float32), y_train[i]) for i in range(len(X_train))]
    test_data = [(X_test[i].astype(np.float32), y_test[i]) for i in range(len(X_test))]
    
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    return trainloader, testloader
    
def load_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader

def load_fashion_mnist():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                                shuffle=True, num_workers=2)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                                shuffle=False, num_workers=2)
    return trainloader, testloader

