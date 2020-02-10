import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FullyConnected(nn.Module):
    # A small fully connected network that takes data like flattened MNIST images
    def __init__(self, input_dim=784):
        super(FullyConnected, self).__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.network(x)
        return out
