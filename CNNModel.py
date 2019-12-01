import torch
import torch.nn as nn
import torch.nn.functional as F

class ECNN_cuda(nn.Module):
    def __init__(self):
        super(ECNN_cuda, self).__init__()
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
        x = self.pool(F.relu(self.conv1(x))).cuda()
        x = self.pool(F.relu(self.conv2(x))).cuda()
        x = x.view(-1,16*16*12).cuda()
        x = F.relu(self.fc1(x)).cuda()
        x = F.relu(self.fc2(x)).cuda()
        x = self.fc3(x).cuda()
        return x

class ECNN_final(nn.Module):
    def __init__(self):
        super(ECNN_final, self).__init__()
        # Input shape shd be of [64,64,3]
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Reduce size to [32,32,6]
        self.conv2= nn.Conv2d(6, 12, 3, padding=1)
        # Reduce size to [16,16,12]
        self.fc1 = nn.Linear(16*16*12, 64)
        #self.fc2 = nn.Linear(256, 64)
        # Since there are 5 classes to identify
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1,16*16*12)

        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ECNN_ini(nn.Module):
    def __init__(self):
        super(ECNN_ini, self).__init__()
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


class ENN_0(nn.Module):
    def __init__(self, num_ftrs):
        super(ENN_0, self).__init__()
        self.num_ftrs = num_ftrs
        print('ENN: Num_ftrs', num_ftrs)
        # Input shape shd be of [64,64,3]
        self.fc1 = nn.Linear(num_ftrs, 256)
        self.fc2 = nn.Linear(256, 64)
        # Reduce size to [16,16,12]
        self.fc3 = nn.Linear(64, 5)
        # Since there are 5 classes to identify

    def forward(self, x):
        x = x.view(-1, self.num_ftrs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ENN_0_4cls(nn.Module):
    def __init__(self, num_ftrs):
        super(ENN_0_4cls, self).__init__()
        self.num_ftrs = num_ftrs
        print('ENN: Num_ftrs', num_ftrs)
        # Input shape shd be of [64,64,3]
        self.fc1 = nn.Linear(num_ftrs, 500)
        self.fc2 = nn.Linear(500, 64)
        # Reduce size to [16,16,12]
        self.fc3 = nn.Linear(64, 4)
        # Since there are 5 classes to identify
        
    def forward(self, x):
        x = x.view(-1, self.num_ftrs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ENN_0_5cls(nn.Module):
    def __init__(self, num_ftrs):
        super(ENN_0_5cls, self).__init__()
        self.num_ftrs = num_ftrs
        print('ENN: Num_ftrs', num_ftrs)
        # Input shape shd be of [64,64,3]
        self.fc1 = nn.Linear(num_ftrs, 512)
        self.fc2 = nn.Linear(512,64)
        # Reduce size to [16,16,12]
        self.fc3 = nn.Linear(64, 5)
        # Since there are 5 classes to identify

    def forward(self, x):
        x = x.view(-1, self.num_ftrs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ENN(nn.Module):
    def __init__(self, num_ftrs):
        super(ENN, self).__init__()
        self.num_ftrs = num_ftrs
        print('ENN: Num_ftrs', num_ftrs)
        # Input shape shd be of [64,64,3]
        self.fc1 = nn.Linear(num_ftrs, 64)
        # Reduce size to [16,16,12]
        self.fc2 = nn.Linear(64, 5)
        # Since there are 5 classes to identify

    def forward(self, x):
        x = x.view(-1, self.num_ftrs)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x

class keras_tl(nn.Module):
    def __init__(self):
        super(keras_tl, self).__init__()
        # Input shape shd be of [64,64,3]
        self.fc1 = nn.Linear(2*2*512, 256)
        self.fc2 = nn.Linear(256, 5)
        # Since there are 5 classes to identify
        #self.fc3 = nn.Linear(64, 5)

    def forward(self, x):

        x = x.view(-1,2*2*512)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x