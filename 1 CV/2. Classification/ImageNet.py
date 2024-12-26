import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(400, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.activation = nn.Sigmoid()
    
    def forward(self, X):
        X = self.activation(self.conv1(X)) # 1×28×28 -> 6×28×28
        X = self.pool(X) # 6×28×28 -> 6×14×14
        X = self.activation(self.conv2(X)) #6×14×14 -> 16×10×10
        X = self.pool(X) # 16×10×10 -> 16×5×5
        X = self.flatten(X) # 16×5×5 -> 1×1×400
        X = self.fc1(X) # 1×1×400 -> 1×1×120
        X = self.fc2(X) # 1×1×120 -> 1×1×84
        X = self.fc3(X) # 1×1×84 -> 1×1×10
        return X


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv = nn.Sequential(
            # conv1 3×227×227 -> 96×55×55
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.ReLU(),
            # maxpool 96×55×55 -> 96×27×27
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv2 96×27×27 -> 256×27×27
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # maxpool 256×27×27 -> 256×13×13
            nn.MaxPool2d(kernel_size=3, stride=2),
            # conv3 256×13×13 -> 384×13×13
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # conv4 384×13×13 -> 256×13×13
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # maxpool 256×13×13 -> 256×6×6
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature)
        return output

class VGG(nn.Module):
    """
    params:
    - conv_arch: a list like [[num1_conv, out1_channels], [num2_conv, out2_channels], ...]
    - num_classes: classes number of output layers
    """    
    def __init__(self, conv_arch, num_classes):
        super().__init__()
        self.activation = nn.ReLU(inplace=True)
        in_channels = 3
        conv_blks = []
        for (num_convs, out_channels) in conv_arch:
            conv_blks.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_blks)
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels * 7 * 7, 4096),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            self.activation,
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def vgg_block(self, num_convs, in_channels, out_channels):
        """
        Define a VGG block.
        Each conv has the same kernel size and feature size.
        The former conv's out_channels = The latter conv's in_channel.
        """
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels,
                                    kernel_size=3, stride=1, padding=1))
            layers.append(self.activation)
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)
    
    def forward(self, img):
        features = self.conv_layers(img)
        output = self.fc_layers(features)
        return output

class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()
    

    def Inception(self):

