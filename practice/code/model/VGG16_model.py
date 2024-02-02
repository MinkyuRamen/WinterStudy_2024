import torch
import torch.nn as nn

class BasicBlock1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(BasicBlock1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.pool(x)

        return x

class BasicBlock2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim):
        super(BasicBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.pool(x)

        return x
    
class vgg16(nn.Module):
    def __init__(self, num_classes=10):
        super(vgg16, self).__init__()
        
        # cifar = 3*32*32 >> 128*2*2
        self.block1 = BasicBlock1(in_channels=3, out_channels=32, hidden_dim=16) # 16*16
        self.block2 = BasicBlock2(in_channels=32, out_channels=128, hidden_dim=64) # 8*8
        self.block3 = BasicBlock2(in_channels=128, out_channels=256, hidden_dim=128) # 4*4

        # classifier
        self.fc1 = nn.Linear(in_features=256*4*4, out_features=2048) # block3의 out_channels=256, 높이4 * 너비4
        self.fc2 = nn.Linear(2048, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x