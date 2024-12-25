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

class AlexNet():
    def __init__(self, num_classes=1000):
        self.conv = nn.Sequential(
            # conv1 3×224×224 -> 96×55×55
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



