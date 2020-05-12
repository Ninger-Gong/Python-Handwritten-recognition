import torch
import numpy as np
import torch.nn as nn
import torchvision
from utils import plot_image, plot_curve
from sklearn import metrics
from sklearn.metrics import confusion_matrix

#load data input
batch_size = 512
num_epochs = 20
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=True, download= True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(   # normalize the data to be distributed around zero, so it could 
                                       (0.1307,), (0.3081, ))          # be zero by chance
                               ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('data', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size, shuffle=False)

# make it iterable
x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), x.max())
plot_image(x, y, 'Real Value')

# construct net work model
class RNN(nn.Module):
    def __init__(self, input, hidden, layer):
        super(RNN, self).__init__()
        self.hidden = hidden
        self.layer = layer
        self.rnn = nn.RNN(input, hidden, layer, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden,10)

    def forward(self, x):
        h0 = torch.zeros(self.layer, x.size(0), self.hidden).requires_grad_()
        out, hn = self.rnn(x, h0.detach())
        out = self.fc(out[:, -1, :])

        return out

# model and optimizer
model = RNN(28, 100, 2)

# loss function and optimizer(loss function has to be decleared before getting into loop)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# train and test
iteration = 0
for epoch in range(num_epochs):
    for step, (images, targets) in enumerate(train_loader):
        images = images.view(-1, 28, 28).requires_grad_()
        optimizer.zero_grad()                 # set gradients for all parameters zero
        outputs = model(images)
        loss = loss_func(outputs, targets)    # use loss function
        loss.backward()
        optimizer.step()

        iteration += 1

        if step % 500 == 0:                    # every 500 batch size, code will go to test process 
            correct = 0
            total_data = 0
            for images, targets in test_loader:
                images = images.view(-1, 28,28)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1) # find the target with the most possibility
                total_data += targets.size(0)
                correct += (predicted == targets).sum()
            accuracy = float(100 * correct / total_data)
            print('Iteration: {}. Loss: {:.3f}%. Accuracy: {:.3f}%'.format(iteration, loss.item(), accuracy))
# produce six random images and show their prediction
data, targets = next(iter(test_loader))
out = RNN(data.view(data.size(0), 28*28),100,2)
pred = out.argmax(dim = 1)
plot_image(data, pred, 'Prediction')
