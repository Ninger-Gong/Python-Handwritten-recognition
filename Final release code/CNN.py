import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import plot_curve, plot_image, image_loader
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

BATCH_SIZE = 512  # How many data working in parallel
Num_Epochs = 20
Device = torch.device("cuda" if torch.cuda.is_available()
                            else "cpu")

# Train dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = True, download = True,
              transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1037,), (0.3081,))  # Move data from the right of zero to be evenly
                                                              # distributed around zero
              ])),
batch_size = BATCH_SIZE, shuffle = True)

# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, download= True,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1037,), (0.3081,))
            ])),
batch_size = BATCH_SIZE, shuffle = True)


# Building model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(p = 0.2)

    def forward(self, x):
        out = self.conv1(x)  # 1* 10 * 24 *24 => 24 = 28 - 5 + 1
        out = F.relu(out)  # non-linear function
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12 => 12 = 24 / 2
        out = self.conv2(out)  # 1* 20 * 10 * 10 => 10 = 12 - 3 + 1
        out = F.relu(out)
        out = out.view(x.size(0), -1)  # 1 * 2000  flatten the tensor to 1-D
        out = self.dropout(F.relu(self.fc1(out)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = F.log_softmax(out, dim=1)
        return out

# Initializing model and optimizer
model = ConvNet().to(Device)
optimizer = optim.Adam(model.parameters())

train_loss = []

# Training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()              # sets gradients of all model parameters to zero
        output = model(data)
        loss = F.nll_loss(output, target)  # loss function
        loss.backward()                    # back-propagation
        optimizer.step()
        train_loss.append(loss.item())

        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{} out of {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# num_correct = []
# Testing function
def test(model, device, test_loader):

    model.eval()  # To turn off Dropout
    test_loss = 0
    correct = 0
    p1 = 0 # Precision Rate
    p2 = 0
    r1 = 0 # Recall Rate
    r2 = 0
    f1 = 0
    with torch.no_grad():  # turn off autograd to save memory
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # Add all losses of the whole batch size
            pred = output.argmax(dim = 1)  # Find the index of the max possibility
            correct += pred.eq(target.view_as(pred)).sum().item()

            confusion_matrix(target, pred)
            p1 = metrics.precision_score(target, pred, average='micro')  # To calculate micro average precision
            p2 = metrics.precision_score(target, pred, average='macro')  # To calculate macro average precision
            r1 = metrics.recall_score(target, pred, average='micro',labels=np.unique(pred))     # Micro average Recall Rate
            r2 = metrics.recall_score(target, pred, average='macro',labels=np.unique(pred))     # Macro average Recall Rate
            f1 = metrics.f1_score(target, pred, average='weighted')

    test_loss /= len(test_loader.dataset)  # Average test loss
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    print("\nPrecision Rate: {:.3f}% and {:.3f}%, Recall Rate: {:.3f}% and {:.3f}%, F1 Score: {:.3f}%\n".format(
        100 * p1, 100 * p2, 100 * r1, 100 * r2, 100 * f1))


start = time.time()
for epoch in range(1, Num_Epochs + 1):
    train(model, Device, train_loader, optimizer, epoch)
    test(model, Device, test_loader)
time_used = (time.time() - start)
minutes = time_used // 60
seconds = time_used - minutes * 60
print("Time consumed: %im %is" % (minutes, seconds))

plot_curve(train_loss)

# total_correct = 0

data, target = next(iter(test_loader))
out = model(data)
pred = out.argmax(dim = 1)
plot_image(data, pred, 'Prediction')









