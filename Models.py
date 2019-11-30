import torch
import torch.nn as nn
import torch.nn.functional as F

class Baseline_64(nn.Module):
    def __init__(self):
        super(Baseline_64, self).__init__()
        self.fc1 = nn.Linear(3*64*64, 3*16*16)
        self.fc2 = nn.Linear(3*16*16, 5)
    def forward(self, x):
        x = x.view(-1, 3*64*64)
        x = F.relu(self.fc1(x))
        return self.fc2(x)