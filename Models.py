import torch
import torch.nn as nn
import torch.nn.functional as f




class Baseline_64(nn.Module):

    def __init__(self):
        super(Baseline_64, self).__init__()

        self.fc1 = nn.Linear(3*64*64, 3*32*32)
        self.fc1 = nn.Linear(3*64*64, 5)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        return f.relu(self.fc2(x))

class Baseline_224(nn.Module):

    def __init__(self):
        super(Baseline_224, self).__init__()

        self.fc1 = nn.Linear(3*224*224, 3*112*112)
        self.fc1 = nn.Linear(3*112*112, 5)

    def forward(self, x):
        x = f.relu(self.fc1(x))
        return f.sigmoid(self.fc2(x))        