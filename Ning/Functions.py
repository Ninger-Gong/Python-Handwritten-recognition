#This file is for the functions of the handwritten digits recognition system project
#Group 43
#Member: Ninger Gong and Bear Xie

import numpy as np
import torch
import torchvision
from torchvision import transforms,datasets
from torch import nn,optim
from torch.utils.data import DataLoader

# judge whether use GPU or not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load the MNIST dataset into the file
mydataset = np.genfromtxt()
transforms = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])

# MNIST dataset
trainset = datasets.MNIST('Train_set',download=True,train=True,transforms = transforms)
testset = datasets.MNIST('test_set',download=True,train=False,transforms = transforms)
#batch size is the number of pictures that we want to read in one go
trainloader = torch.utils.data.Dataloader(trainset,batch_size = 64,shuffle = True)
testloader = torch.utils.data.Dataloader(testset,batch_size = 64,shuffle = True)

#single picture viewable
image,label = next(iter(trainloader))
img = torchvision.utils.make_grid(image)
img = img.numpy().transpose(1,2,0)
std = [0.5,0.5,0.5]
mean = [0.5,0.5,0.5]
img = img*std+mean
print(label)



#Models
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.convolution = nn.Sequential(  # the first convolution layer
            nn.Conv2d(1, 64, kernel_size=3,stride=1,padding=1),  # 25*26*26 or (1,64,kernel_size=3,stride = 1,padding = 1)
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2,kernel_size=2),
            ##nn.BatchNorm2d(25),
            ##nn.ReLU(inplace=True)
        )
        self.dense = nn.Sequential(
            nn.Linear(14*14*128,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,10)
        )

    def forward(self,x):
        x = self.convolution(x)
        x = x.view(-1,14*14*128)
        x = self.dense(x)
        return x

#batch_size = 64
#LR =0.001

model = Model().to(device)
optimizer = optim.Adam( #improve the algorithm
    model.parameter(),
)
criterion = nn.CrossEntropyLoss()

#training model
def train_Model(model,device,trainloader,optimizer,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device),target.to(device)
        optimizer.zero_grad()
        output = model(data)
        optimizer.step()
        loss = criterion(output,target)
        loss.backward()
        if batch_idx % 10 == 0:
            Loss.append = loss.data[0]
        
    
# testing model
def testing_Model(model,device,testloader,optimizer,epoch):
    model.eval()
    

# print out the final result that is recognized
def inputImage(filename):
    img = Image.open(filename)
    img = np.array(img)
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    array1 = []
    for i in range(28):
        for j in range(28):
            pixel = 1.0 - float(img.getpixel((j, i))) / 255
            array1.append(pixel)
    array = np.array(array1).reshape((1, 28, 28, 1))
    return array


# print the predict number
for i in range(20):
    training_batch = zip(range(0, len(trainloader), batch_size))
    for start, end in training_batch:
        

