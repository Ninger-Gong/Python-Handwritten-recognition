#AlexNet & MNIST


import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

#Net Model

#Define the Network Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super( AlexNet, self ).__init__()
        self.features = nn.Sequential(
            # because the size of images of MNIST is 28x28， but AlexNet is 227x227. So the parameter of the Net needs to be changed
            # First convolution layer，1 input channel，32 output with 11*11 kernel
            #block one
            nn.Conv2d( 1, 32, kernel_size=3, padding=1 ),#AlexCONV1(3,96, k=11,s=4,p=0)
            # relu after convolition
            nn.ReLU( inplace=True ),
            # Max poolling
            nn.MaxPool2d( kernel_size=2, stride=2 ),#AlexPool1(k=3, s=2)
            
            #block two
            nn.Conv2d( 32, 64, kernel_size=3,stride=1, padding=1 ),#AlexCONV2(96, 256,k=5,s=1,p=2)
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( kernel_size=2, stride=2 ),#AlexPool2(k=3,s=2)

            #block three
            nn.Conv2d( 64, 128, kernel_size=3, stride= 1, padding=1 ),#AlexCONV3(256,384,k=3,s=1,p=1)
            nn.ReLU( inplace=True ),

            #block four
            nn.Conv2d( 128, 256, kernel_size=3,stride= 1, padding=1 ),#AlexCONV4(384, 384, k=3,s=1,p=1)
            nn.ReLU( inplace=True ),

            #block five
            nn.Conv2d( 256, 256, kernel_size=3,stride= 1, padding=1 ),#AlexCONV5(384, 256, k=3, s=1,p=1)
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( kernel_size=2, stride=2 ),
        )
        # full-connection layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear( 256 * 3 * 3, 1024 ),#AlexFC6(256*6*6, 4096)
            nn.ReLU( inplace=True ),
            nn.Dropout(),
            nn.Linear( 1024, 512 ),#AlexFC6(4096,4096)
            nn.ReLU( inplace=True ),
            nn.Linear( 512, 10 ),#AlexFC6(4096,1000)
        )

    def forward(self, x):
        x = self.features( x )
        x = x.view( -1, 256 * 3 * 3 )#Alex: x = x.view(-1, 256*6*6)
        x = self.classifier( x )
        return x

#transform
transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(), #if the dataset is 3D_RGB it need to set to 2D
                    transforms.ToTensor(), #set the dataset from images to tensor


])

transform1 = transforms.Compose([
                    transforms.ToTensor()
])

# dataset of MNIST
#trainset = torchvision.datasets.MNIST(root='./data',train=True,download=True,transform=transform)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,shuffle=True,num_workers=0)

#testset = torchvision.datasets.MNIST(root='./data',train=False,download=True,transform=transform1)
#testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False,num_workers=0)

# load the data MNIST
# train loader
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = True, download = True,
              transform = transform),
    batch_size = 100, shuffle = True)

# test loader
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, download = True, transform = transform1),
    batch_size = 100, shuffle = True)

#net
net = AlexNet()

#loss function
criterion = nn.CrossEntropyLoss()

#SGD
#optimizer = optim.SGD(net.parameters(),lr=1e-3, momentum=0.9)
#Adam
optimizer = optim.Adam(model.parameters())

#device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)

print("Start Training!")

num_epochs = 20 #number of training

for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f'%(epoch+1, (i+1)*100, loss.item()))

print("Finished Traning")


#save the model
torch.save(net, 'MNIST.pkl')
net = torch.load('MNIST.pkl')

#start recognition
with torch.no_grad():
    #All Tensor requires_grad will be set as False
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:{}%'.format(100 * correct / total)) #Accuracy of the system
