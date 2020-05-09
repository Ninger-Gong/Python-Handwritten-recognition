import torch.nn as nn
import torch
from torch.nn import functional as F

class ResidualBlock( nn.Module ):
    # module：Residual Block
    def __init__(self, in_ch, out_ch, stride=1, shortcut=None):
        super( ResidualBlock, self ).__init__()
        self.left = nn.Sequential(
            nn.Conv2d( in_ch, out_ch, 3, stride, padding=1, bias=False ),
            nn.BatchNorm2d( out_ch ),
            nn.ReLU( inplace=True ),
            nn.Conv2d( out_ch, out_ch, 3, stride=1, padding=1, bias=False ),
            nn.BatchNorm2d( out_ch )
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left( x )
        residual = x if self.right is None else self.right( x )
        out += residual
        return F.relu( out )


class ResNet( nn.Module ):  # 224x224x3
    # module:ResNet
    def __init__(self, num_classes=1):
        super( ResNet, self ).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d( 3, 64, 7, stride=2, padding=3, bias=False ),  # (224+2*p-)/2(向下取整)+1，size减半->112
            nn.BatchNorm2d( 64 ),  # 112x112x64
            nn.ReLU( inplace=True ),
            nn.MaxPool2d( 3, 2, 1 )  # kernel_size=3, stride=2, padding=1
        )  # 56x56x64

        # repeated layer,having 3,4,6,3 residual blocks
        self.layer1 = self.make_layer( 64, 64, 3 )  # 56x56x64
        self.layer2 = self.make_layer( 64, 128, 4, stride=2 )  # stride=2,28x28x128
        self.layer3 = self.make_layer( 128, 256, 6, stride=2 )  # 14x14x256
        self.layer4 = self.make_layer( 256, 512, 3, stride=2 )  # 7x7x512
        # full connect
        self.dropout = nn.Dropout( p=0.2 )
        self.fc = nn.Linear( 512, num_classes )

    def make_layer(self, in_ch, out_ch, block_num, stride=1):
        #
        shortcut = nn.Sequential(  # 
            nn.Conv2d( in_ch, out_ch, 1, stride, bias=False ),  
            nn.BatchNorm2d( out_ch )
        )
        layers = []
        layers.append( ResidualBlock( in_ch, out_ch, stride, shortcut ) )

        for i in range( 1, block_num ):
            layers.append( ResidualBlock( out_ch, out_ch ) ) 
        return nn.Sequential( *layers )

    def forward(self, x):  # 224x224x3
        x = self.pre( x )  # 56x56x64
        x = self.layer1( x )  # 56x56x64
        x = self.layer2( x )  # 28x28x128
        x = self.layer3( x )  # 14x14x256
        x = self.layer4( x )  # 7x7x512
        x = F.avg_pool2d( x, 7 )  # 1x1x512
        x = x.view( x.size( 0 ), -1 )  # 将输出拉伸为一行：1x512
        x = self.fc( x )  # 1x1
        out = self.dropout(F.relu(self.fc(x)))
        # nn.BCELoss:二分类用的交叉熵，用的时候需要在该层前面加上 Sigmoid 函数
        out = F.log_softmax( out, dim=1 )
        return (out)
