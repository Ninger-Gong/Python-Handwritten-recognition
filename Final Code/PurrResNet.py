import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
#from utils import plot_curve, plot_image
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from tqdm.autonotebook import tqdm

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color = 'blue')
    plt.legend(['value'], loc = 'upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()


def plot_image(img, label, name):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(img[i][0]*0.3081+0.1307, cmap='gray', interpolation= 'none')
        plt.title("{}: {}".format(name, label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def one_hot(label, depth = 10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim = 1, index = idx, value = 1)
    return out


# Visualize some samples
def visualize(test_loader):
    batch = next(iter(test_loader))
    samples = batch[0][:5]
    y_true = batch[1]
    for i, sample in enumerate(samples):
        plt.subplot(1, 5, i+1)
        plt.title('Numbers: %i' % y_true[i])
        plt.imshow(sample.numpy().reshape((28, 28)))
        plt.axis('off')

BATCH_SIZE = 512
Num_Epochs = 3
Device = torch.device("cuda" if torch.cuda.is_available()
                            else "cpu")


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = True, download = True,
              transform = transforms.Compose([
                  transforms.Resize(224),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5))
              ])),
    batch_size = BATCH_SIZE, shuffle = True)

# Test dataset
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])),
    batch_size = BATCH_SIZE, shuffle = True)



class ResidualBlock( nn.Module ):
    # module: Residual Block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super( ResidualBlock, self ).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, 1, bias=False ),
            nn.BatchNorm2d( outchannel ),
            nn.ReLU( inplace=True ),
            nn.Conv2d( outchannel, outchannel, 1, 1, 1, bias=False ),
            nn.BatchNorm2d( outchannel )
        )

        self.right = shortcut

    def forward(self, x):
        out = self.left( x )
        residual = x if self.right is None else self.right( x )
        out += residual
        return F.relu( out )


class ResNet( nn.Module ):
    # Main module:ResNet
    # ResNet with multiple layers,each layer contains with multiple residual block
    # residual block , use _make_layer for layers
    def __init__(self, num_classes=10):
        super( ResNet, self ).__init__()
        self.pretreat = nn.Conv2d(1,3,kernel_size=1) # turn the dataset from 1D to 3D
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),# 1 input tunnels, 64 output tunnels, convolutional layer 7*7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # repeated layer,with 3,4,6,3 residual blocks each
        self.layer1 = self._make_layer( 64, 64, 3 )
        self.layer2 = self._make_layer( 64, 128, 4, stride=2 )
        self.layer3 = self._make_layer( 128, 256, 6, stride=2 )
        self.layer4 = self._make_layer( 256, 512, 3, stride=2 )

        # full connection for classify
        self.fc = nn.Linear( 512, num_classes )

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        # buiding layer,include multiple residual block
        shortcut = nn.Sequential(
            nn.Conv2d( inchannel, outchannel, 1, stride, bias=False ),
            nn.BatchNorm2d( outchannel ) )

        layers = []
        layers.append( ResidualBlock( inchannel, outchannel, stride, shortcut ) )

        for i in range( 1, block_num ):
            layers.append( ResidualBlock( outchannel, outchannel ) )
        return nn.Sequential( *layers )

    def forward(self, x):
        x = self.pretreat(x)
        x = self.pre( x )
        x = self.layer1( x )
        x = self.layer2( x )
        x = self.layer3( x )
        x = self.layer4( x )

        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0),-1)
        out = self.fc(x)
        return out

# Initializing model and optimizer
model = ResNet().to(Device)
optimizer = optim.Adam(model.parameters())
train_loss = []
# Training function
def train(model, device, train_loader, optimizer, epoch):
    start = time.time()
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
        time_used = (time.time() - start)
        minutes = time_used // 60
        seconds = time_used - minutes * 60
        if (batch_idx + 1) % 30 == 0:
            print("Time consumed: %im %is" % (minutes, seconds))
            print('Train Epoch: {} [{} out of {} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# num_correct = []
# Testing function
def test(model, device, test_loader):

    model.eval()  # To shut down Dropout
    test_loss = 0
    correct = 0
    p1 = 0 # Precision Rate
    p2 = 0
    r1 = 0 # Recall Rate
    r2 = 0
    f1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # Add all losses of the whole batch size
            pred = output.argmax(dim = 1)  # Find the index of max possibility
            correct += pred.eq(target.view_as(pred)).sum().item()

            confusion_matrix(target,pred)
            p1 = metrics.precision_score(target, pred, average='micro')  # To calculate micro average precision
            p2 = metrics.precision_score(target, pred, average='macro')  # To calculate macro average precision
            r1 = metrics.recall_score(target, pred, average='micro')     # Micro average Recall Rate
            r2 = metrics.recall_score(target, pred, average='macro')     # Macro average Recall Rate
            f1 = metrics.f1_score(target, pred, average='weighted')

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))
    print("\nPrecision Rate: {:.3f}% and {:.3f}%, Recall Rate: {:.3f}% and {:.3f}%, F1 Score: {:.3f}%\n".format(
        100 * p1, 100 * p2, 100 * r1, 100 * r2, 100 * f1))



for epoch in range(1, Num_Epochs + 1):
    train(model, Device, train_loader, optimizer, epoch)
    test(model, Device, test_loader)

plot_curve(train_loss)

total_correct = 0


data, target = next(iter(test_loader))
out = model(data)
pred = out.argmax(dim = 1)
plot_image(data, pred, 'test')

