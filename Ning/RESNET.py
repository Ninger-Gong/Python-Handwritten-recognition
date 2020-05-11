from torch import nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv3x3( self.inplanes, planes * block.expansion, stride ),
                nn.BatchNorm2d( planes * block.expansion ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample ) )
        self.inplanes = planes * block.expansion
        for _ in range( 1, blocks ):
            layers.append( block( self.inplanes, planes ) )

        return nn.Sequential(*layers)

class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self,inplane,planes,stride = 1,downsample = None):
        super(ResidualBlock,self).__init__()
        self.conv1 = conv3x3(inplane,planes,stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes,planes)
        self.bn2 = nn.BatchNorm2d(planes)
        #self.conv3 = conv3x3( planes, planes )
        #self.bn3 = nn.BatchNorm2d( planes )
    
        self.downsample = downsample
        self.stride = stride

    def forward(self,x):
        inden = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        #out = self.conv3(out)
        #out = self.bn3(out)
        #out = self.relu(out)

        if self.downsample() is not None:
            inden = self.downsample(x)

        out += inden
        out = self.relu(out)

        return out
        
        
        
net_args = {
    "block": ResidualBlock,
    "layers": [2, 2, 2, 2]
    }
model = ResNet(**net_args).to(DEVICE)
