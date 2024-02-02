import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self,x):
        # skip connection을 위한 초기 initial input
        x_ = x

        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        # skip connection
        x_ = self.downsample(x_)
        x += x_
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        self.b1 = BasicBlock(3, 64)
        self.b2 = BasicBlock(64, 128)
        self.b3 = BasicBlock(128, 256)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, num_classes)

        self.relu = nn.ReLU()
    
    def forward(self, x): # x = 3*32*32
        x = self.b1(x) # 64*32*32
        x = self.pool(x) # 64*16*16

        x = self.b2(x) # 128*16*16
        x = self.pool(x) # 128*8*8

        x = self.b3(x) # 256*8*8
        x = self.pool(x) # 256*4*4

        x = torch.flatten(x, start_dim=1) # 1*4096
        x = self.fc1(x) # 1*2048
        x = self.relu(x)
        x = self.fc2(x) # 1*512
        x = self.relu(x)
        x = self.fc3(x) # 1*10

        return x