import torch
import torch.nn as nn
from torch.nn import functional as F

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()

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
        super(AlexNet, self).__init__()
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
        super(VGG, self).__init__()
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
    def __init__(self, img_channles=3, num_classes=10):
        super(GoogleNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channles, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64,192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(1024, num_classes)

    def InceptionV1(self, X, c1, c2, c3, c4):
        """
        params:
        - X: input of Inception block (batch_size, channels, length, width)
        - c1~c4: deciding channels in Inception
        """
        in_channels = X.shape[1]
        path1 = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1),
            nn.ReLU()
        )
        path2 = nn.Sequential(
            nn.Conv2d(in_channels, c2[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1),
            nn.ReLU()
        )
        path3 = nn.Sequential(
            nn.Conv2d(in_channels, c3[0], kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2),
            nn.ReLU()
        )
        path4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, c4, kernel_size=1),
            nn.ReLU()
        )
        X1, X2, X3, X4 = path1(X), path2(X), path3(X), path4(X)
        # Xi shape:(batch, channel, length, width) by 'dim=1' we get overlapped channels
        return torch.cat((X1, X2, X3, X4), dim=1)
    
    def forward(self, img):
        features = self.conv(img)
        features = self.InceptionV1(features, 64, (96, 128), (16, 32), 32)
        features = self.InceptionV1(features, 128, (128, 192), (32, 96), 64)
        features = self.maxpool(features)
        features = self.InceptionV1(features, 192, (96, 208), (16, 48), 64)
        features = self.InceptionV1(features, 160, (112, 224), (24, 64), 64)
        features = self.InceptionV1(features, 128, (128, 256), (24, 64), 64)
        features = self.InceptionV1(features, 112, (144, 288), (32, 64), 64)
        features = self.InceptionV1(features, 256, (160, 320), (32, 128), 128)
        features = self.maxpool(features)
        features = self.InceptionV1(features, 256, (160, 320), (32, 128), 128)
        features = self.InceptionV1(features, 384, (192, 384), (48, 128), 128)
        features = self.avgpool(features)
        features = self.flatten(features)
        output = self.linear(features)
        return output
