import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    # 前面1x1和3x3卷积的filter个数相等，最后1x1卷积是其expansion倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet( nn.Module ):
    def __init__(self, block, num_blocks, num_classes=10):
        super( ResNet, self ).__init__()
        self.in_planes = 64
        self.pre = nn.Conv2d(1,3,kernel_size=1)

        self.conv1 = nn.Conv2d( 3, 64, kernel_size=3,
                                stride=1, padding=1, bias=False )
        self.bn1 = nn.BatchNorm2d( 64 )

        self.layer1 = self._make_layer( block, 64, num_blocks[0], stride=1 )
        self.layer2 = self._make_layer( block, 128, num_blocks[1], stride=2 )
        self.layer3 = self._make_layer( block, 256, num_blocks[2], stride=2 )
        self.layer4 = self._make_layer( block, 512, num_blocks[3], stride=2 )
        self.linear = nn.Linear( 512 * block.expansion, num_classes )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append( block( self.in_planes, planes, stride ) )
            self.in_planes = planes * block.expansion
        return nn.Sequential( *layers )

    def forward(self, x):
        x = self.pre(x)
        out = F.relu( self.bn1( self.conv1( x ) ) )
        out = self.layer1( out )
        out = self.layer2( out )
        out = self.layer3( out )
        out = self.layer4( out )
        out = F.avg_pool2d( out, 4 )
        out = out.view( out.size( 0 ), -1 )
        out = self.linear( out )
        return out

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

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

model = ResNet50().to(Device)

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
