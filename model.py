import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import sigmoid
from torch_geometric.nn import ResGatedGraphConv, GATv2Conv, ChebConv, TAGConv, SAGEConv, GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.utils import grid
from mygat import Att
IMG_SIZE = 28
OUT = 10
featurelength = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")     

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = Att(IMG_SIZE * IMG_SIZE, featurelength * 2, heads = 4)
        
        self.fc1 = nn.Linear(featurelength * 4 , OUT)
        
        self.dropout = nn.Dropout(0.15)
        self.pool = nn.MaxPool1d(2)

    def forward(self, x, train):
        batch_size = x.shape[0]
        x_1 = torch.reshape(x, (batch_size, IMG_SIZE * IMG_SIZE))

        x_1 = self.conv1(x_1)
        x_1 = F.relu(x_1)
         
        if train:
            x_1 = self.dropout(x_1)
        x_2 = self.pool(x_1.unsqueeze(0)).squeeze(0)

        
        attention = torch.matmul(x_2, torch.transpose(x_2, 0, 1))
        attention = sigmoid(attention)
        attention = attention / torch.sum(attention, dim=1).unsqueeze(1)
        
        x_3 = torch.matmul(attention, x_2)
        
        residual = x_3 + x_2
        
        x_4 = self.fc1(residual)
        return x_4