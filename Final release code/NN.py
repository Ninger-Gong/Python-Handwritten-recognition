import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from torch import optim
import torchvision
from matplotlib import pyplot as plt
from utils import plot_image, plot_curve, one_hot
from sklearn import metrics
from sklearn.metrics import confusion_matrix


batch_size = 512
num_epochs = 2
# load dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=True, download= True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081, ))
                               ])),
    batch_size = batch_size, shuffle = True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)
x, y = next(iter(train_loader))
print(x.shape, y.shape,x.min(), x.max())
plot_image(x, y ,'Real Value')

# construct model
class NormalNet(nn.Module):
    def __init__(self):
        super(NormalNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

NormalNet = NormalNet()
optimizer = optim.SGD(NormalNet.parameters(), lr=0.01, momentum= 0.9)

# train process
train_loss = []
for epoch in range (1,num_epochs+1):
    for batch_idx, (data,targets) in enumerate(train_loader):

        data = data.view(data.size(0), 28*28)  # flatten 4-dimension to 2-dimension
        out = NormalNet(data)
        targets_onehot = one_hot(targets)
        loss = F.mse_loss(out, targets_onehot)  # Mean squared error
        optimizer.zero_grad() # set gradients for all parameters zero
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        if batch_idx % 512 == 0:
            print('Number of Epochs: {} and loss is {:.5f}'.format(epoch, loss.item()))

plot_curve(train_loss)



# test process
num_correct = 0
for data, target in test_loader:
    data = data.view(data.size(0), 28*28)
    out = NormalNet(data)
    pred = out.argmax(dim = 1)
    correct = pred.eq(target).sum().float().item()
    num_correct += correct

    confusion_matrix(target, pred)
    p1 = metrics.precision_score(target, pred, average='micro')  # To calculate micro average precision
    p2 = metrics.precision_score(target, pred, average='macro')  # To calculate macro average precision
    r1 = metrics.recall_score(target, pred, average='micro', labels=np.unique(pred))  # Micro average Recall Rate
    r2 = metrics.recall_score(target, pred, average='macro', labels=np.unique(pred))  # Macro average Recall Rate
    f1 = metrics.f1_score(target, pred, average='weighted')

print("\nPrecision Rate: {:.3f}% and {:.3f}%, Recall Rate: {:.3f}% and {:.3f}%, F1 Score: {:.3f}%\n".format(
        100 * p1, 100 * p2, 100 * r1, 100 * r2, 100 * f1))
acc = num_correct / len(test_loader.dataset)
print('Test Accuracy: {:.3f}%'.format(acc))

data, targets = next(iter(test_loader))
out = NormalNet(data.view(data.size(0), 28*28))
pred = out.argmax(dim = 1)
plot_image(data, pred, 'Prediction')
