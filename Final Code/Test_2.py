import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 512 # 大概需要2G的显存
EPOCHS = 10 # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 下载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = True, download = True,
              transform = transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Normalize((0.1037,), (0.3081,))
              ])),
    batch_size = BATCH_SIZE, shuffle = True)

# 测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train = False, transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1037,), (0.3081,))
    ])),
    batch_size = BATCH_SIZE, shuffle = True)


# 定义模型
class LeNet( nn.Module ):
    def __init__(self):
        super( LeNet, self ).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d( 1, 6, 3, 1, 2 ),
            nn.ReLU(),
            nn.MaxPool2d( 2, 2 )
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d( 6, 16, 5 ),
            nn.ReLU(),
            nn.MaxPool2d( 2, 2 )
        )

        self.fc1 = nn.Sequential(
            nn.Linear( 16 * 5 * 5, 120 ),
            nn.BatchNorm1d( 120 ),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear( 120, 84 ),
            nn.BatchNorm1d( 84 ),
            nn.ReLU(),
            nn.Linear( 84, 10 )
        )

    def forward(self, x):
        x = self.conv1( x )
        x = self.conv2( x )
        x = x.view( x.size()[0], -1 )
        x = self.fc1( x )
        x = self.fc2( x )
        return x

    # 生成模型和优化器
model = LeNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

for epoch in range(1, EPOCHS + 1):
    train(model,  DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)


#def view_classify(img, ps):
#    ''' Function for viewing an image and it's predicted classes.
#    '''
#    ps = ps.data.numpy().squeeze()

#    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
#    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
#    ax1.axis('off')
#    ax2.barh(np.arange(10), ps)
#    ax2.set_aspect(0.1)
    #   ax2.set_yticks(np.arange(10))
    #   ax2.set_yticklabels(np.arange(10))
    #    ax2.set_title('Class Probability')
    #    ax2.set_xlim(0, 1.1)
#    plt.tight_layout()

#images, labels = next(iter(test_loader))

# input the image
#img = Image.open("Digit6.png").convert("L")
#img = img.resize((28,28))
# Turn off gradients to speed up this part
#with torch.no_grad():
#    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
#ps = torch.exp(logps)
#probab = list(ps.numpy()[0])
#print("Predicted Digit =", probab.index(max(probab)))
#view_classify(img.view(1, 28, 28), ps)
