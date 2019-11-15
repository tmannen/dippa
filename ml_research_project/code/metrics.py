import argparse
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from torch.optim.lr_scheduler import MultiStepLR

from model import NetworkNvidia
from dataloading import CARLADataset, Flip, ToTensor
from torch.utils.data import Dataset, DataLoader
from train import train

def get_metrics(model, data):
