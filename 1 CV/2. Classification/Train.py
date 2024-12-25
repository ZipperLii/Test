import torch
from ImageNet import AlexNet

X = torch.randn((20, 3, 227, 227))
net = AlexNet()
print(net.forward(X).shape)