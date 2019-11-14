import torch
import torch.nn as nn
import torch.nn.functional as F

class ECNN(nn.Module):
    def __init__(self):
        super(ECNN, self).__init__()
        # Input shape shd be of [64,64,3]
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduce size to [32,32,6]
        self.conv2= nn.Conv2d(6, 12, 3, padding=1)
        # Reduce size to [16,16,12]
        self.fc1 = nn.Linear(16*16*12, 256)
        self.fc2 = nn.Linear(256, 64)
        # Since there are 5 classes to identify
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,16*16*12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

