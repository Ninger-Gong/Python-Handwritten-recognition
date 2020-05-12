##This file is for the COMPSYS302 Python Project
# Group 43
# Ninger Gong and Bear Xie

import torch
import torchvision
from matplotlib import transforms
from torch import nn
import numpy as np

## load the MNIST dataset into the file
train_dataset = torchvision.datasets.MNIST(root='./data',train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='./data',train=False, transform=transforms.ToTensor(),download = True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=100, shuffle=False)

# each convulutional layer is following a max pooling layer
class cnn(nn.Module):
    def __init__(self):
        super(cnn,self).__init__()
        self.layer1 = nn.Sequential(#the first convolution layer
            nn.Conv2d(1,25,kernel_size=3), #25*26*26
            nn.BatchNorm2d(25),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential( #the first Max pooling layer
            nn.MaxPool2d(kernel_size=2,stride=2) #25*13*13
        )
        self.layer3 = nn.Sequential(# the second convolution layer
            nn.Conv2d(25, 50, kernel_size=3), #50*3*3
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential( # the second max pooing layer
            nn.MaxPool2d(kernel_size=2,stride=2) #50*5*5
        )
        self.layer5 = nn.Sequential( #the FC layer
            nn.Linear(50*5*5,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = self.layer5(x)
        return x


