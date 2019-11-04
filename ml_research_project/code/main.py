"""
Main pipeline of Self-driving car training.
@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import MultiStepLR

from model import NetworkNvidia
from dataloading import CARLADataset
from torch.utils.data import Dataset, DataLoader
from train import train

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description='Main pipeline for self-driving vehicles simulation using machine learning.')
    # directory
    parser.add_argument('--dataroot', type=str, default="../data/", help='path to dataset')
    args = parser.parse_args()

    return args

def main():
    """Main pipeline."""
    # parse command line arguments
    # load trainig set and split
    args = parse_args()
    dataset = CARLADataset(args.dataroot)
    print("==> Preparing dataset ...")
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

    # define model
    print("==> Initialize model ...")
    model = NetworkNvidia().cuda()

    # define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)

    # training
    print("==> Start training ...")
    train(model, criterion, optimizer, scheduler, dataloader)


if __name__ == '__main__':
    main()