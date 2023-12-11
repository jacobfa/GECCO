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

    train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)

    return train_dataloader, test_dataloader

def load_coco():
    X_train = torch.load('/data3/jacob/COCO/X_train_128.pt')
    X_test = torch.load('/data3/jacob/COCO/X_val_128.pt')
    y_train = torch.load('/data3/jacob/COCO/y_train.pt')
    y_test = torch.load('/data3/jacob/COCO/y_val.pt')
    
    # make X_train, X_test grayscale
    # X_train = X_train[:,0,:,:]
    # X_test = X_test[:,0,:,:]
    
    trainloader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
    testloader = DataLoader(list(zip(X_test, y_test)), batch_size=64, shuffle=False)
    
    return trainloader, testloader

def load_imagenet():
    from torchvision.datasets import ImageNet
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/imagenet/train'
    valdir = '/data3/jacob/imagenet/val'
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    train_dataset = ImageNet(
        split='train',
        root=traindir,
        transform=transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        ImageNet(
            split='val',
            root=valdir,
            transform=transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader

def load_cifar10():
    from torchvision.datasets import CIFAR10
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/cifar10'
    valdir = '/data3/jacob/cifar10'

    train_dataset = CIFAR10(
        root=traindir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        CIFAR10(
            root=valdir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader

def load_cifar100():
    from torchvision.datasets import CIFAR100
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/cifar100'
    valdir = '/data3/jacob/cifar100'

    train_dataset = CIFAR100(
        root=traindir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        CIFAR100(
            root=valdir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader

def load_imagenet():
    import numpy as np
    import torch
    import os
    
    # X_train = []
    # y_train = []
    # X_val = []
    # y_val = []
    
    # path1 = '/data3/jacob/imagenet/Imagenet64_train_part1_npz/'
    # path2 = '/data3/jacob/imagenet/Imagenet64_train_part2_npz/'
    # path_val = '/data3/jacob/imagenet/Imagenet64_val_npz/'
    
    # for file in os.listdir(path1):
    #     data = np.load(path1 + file)
    #     X_train.append(data['data'])
    #     y_train.append(data['labels'])
    
    # for file in os.listdir(path2):
    #     data = np.load(path2 + file)
    #     X_train.append(data['data'])
    #     y_train.append(data['labels'])
        
    # for file in os.listdir(path_val):
    #     data = np.load(path_val + file)
    #     X_val.append(data['data'])
    #     y_val.append(data['labels'])
        
    # X_train = np.concatenate(X_train, axis=0)
    # y_train = np.concatenate(y_train, axis=0)
    # X_val = np.concatenate(X_val, axis=0)
    # y_val = np.concatenate(y_val, axis=0)
    
    # X_train = torch.from_numpy(X_train).to(torch.float32)
    # X_val = torch.from_numpy(X_val).to(torch.float32)
    # y_train = torch.from_numpy(y_train).to(torch.int64)
    # y_val = torch.from_numpy(y_val).to(torch.int64)
    
    # X_train = torch.reshape(X_train, (X_train.shape[0], 3, 64, 64))
    # X_val = torch.reshape(X_val, (X_val.shape[0], 3, 64, 64))
    # y_train = y_train - 1
    # y_val = y_val - 1
        
    # trainloader = DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
    # testloader = DataLoader(list(zip(X_val, y_val)), batch_size=64, shuffle=False)        
    
    # torch.save(trainloader, '/data3/jacob/imagenet/trainloader.pt')
    # torch.save(testloader, '/data3/jacob/imagenet/testloader.pt')
    
    trainloader = torch.load('/data3/jacob/imagenet/trainloader.pt')
    testloader = torch.load('/data3/jacob/imagenet/testloader.pt')
    
    
    return trainloader, testloader


def load_stl10():
    from torchvision.datasets import STL10
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/stl10'
    valdir = '/data3/jacob/stl10'

    train_dataset = STL10(
        root=traindir,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        STL10(
            root=valdir,
            split='test',
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader

def load_caltech256():
    from torchvision.datasets import Caltech256
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/caltech256'
    valdir = '/data3/jacob/caltech256'

    # resize collate function
    def my_collate(batch):
        data = [item[0] for item in batch]
        target = [item[1] for item in batch]
        target = torch.LongTensor(target)
        return [data, target]
    
    train_dataset = Caltech256(
        root=traindir,
        download=True,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor()
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=my_collate)
    
    val_loader = DataLoader(
        Caltech256(
            root=valdir,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Lambda(lambda x: x.convert('RGB')),
                transforms.ToTensor()
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=my_collate)

    
    return train_loader, val_loader

def load_cifar10_contrastive():
    from torchvision.datasets import CIFAR10
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/cifar10'
    valdir = '/data3/jacob/cifar10'

    transforms = T.Compose([
        # contrastive learning
        T.RandomResizedCrop(32),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        T.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
        T.ToTensor(),
    ])
        
    train_dataset = CIFAR10(
        root=traindir,
        train=True,
        download=True,
        transform=transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        CIFAR10(
            root=valdir,
            train=False,
            download=True,
            transform=transforms),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader

def load_merced():
    import numpy as np
    import torch
    from torch_geometric.loader import DataLoader
    # transforms
    from torchvision import transforms as T
    import os
    from PIL import Image
    merced_dir = '/data3/jacob/UCMerced/UCMerced_LandUse/Images/'
    # each sub directory is a class
    class_names = sorted(os.listdir(merced_dir))
    # each file in a sub directory is an image
    images = []
    labels = []
    # tif to image transform to tensor
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize( (256, 256) ),
        T.ToTensor()
    ])
    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(merced_dir, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = transform(np.array(Image.open(image_path)))
            images.append(image)
            labels.append(i)
            
    split = int(len(images) * 0.8)
    train_images, train_labels = images[:split], labels[:split]
    test_images, test_labels = images[split:], labels[split:]
    
    train = list(zip(train_images, train_labels))
    test = list(zip(test_images, test_labels))
    
    trainloader, testloader = DataLoader(train, batch_size=64, shuffle=True), DataLoader(test, batch_size=64, shuffle=True)
    return trainloader, testloader

def load_mnist():
    from torchvision.datasets import MNIST
    from torchvision import transforms
    from torch.utils.data import DataLoader
    import torch
    
    # Data loading code
    traindir = '/data3/jacob/mnist'
    valdir = '/data3/jacob/mnist'

    train_dataset = MNIST(
        root=traindir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))
    
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True,
        num_workers=4, pin_memory=True)
    
    val_loader = DataLoader(
        MNIST(
            root=valdir,
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])),
        batch_size=64, shuffle=False,
        num_workers=4, pin_memory=True)
    
    
    return train_loader, val_loader