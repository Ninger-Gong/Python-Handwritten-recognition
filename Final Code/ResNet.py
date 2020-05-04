#ResNet
class ResidualBlock(nn.Module):
    ### residual block
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        ### Convovutional
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.right = shortcut

    def forward(self, x):
        ### constant isolation map first，then plus the out of convolution then ReLU
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return torch.nn.functional.relu(out)

class ResNet(torch.nn.Module):
    def __init__(self,num_class = 1000):
        super(ResNet, self).__init__()
        # Do the 7*7 Convolution first
        self.pre = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False), # 3 input tunnels, 64 output tunnels, convolutional layer 7*7
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,1,1) # in channel, out channel, padding
        )
        # 32 layers in total
        self.layer1 = self
._make_layer(64,128,3)
        self.layer2 = self._make_layer(128,256,4,stride = 2)
        self.layer3 = self._make_layer(256,512,6,stride = 2)
        self.layer4 = self._make_layer(512,512,3, stride = 2)
        #FC layer
        self.fc = nn.Linear(512, num_class)

    def _make_layer(self,inchannel,outchannel,block_num,stride = 1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))

        for i in range(1,block_num+1):
            layers.append(
                ResidualBlock(outchannel,outchannel)
            )
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.pre
        x = self.layer1
        x = self.layer2
        x = self.layer3
        x = self.layer4

        x = torch.nn.functional.avg_pool2d(x,2) # feature map to one feature

        x = x.view(x.size(0),-1)

        return self.fc(x)

    
