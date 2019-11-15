"""
Main pipeline of Self-driving car training.
@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316

Modified by tmannen
"""

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

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(description='Main pipeline for self-driving vehicles simulation using machine learning.')
    # directory
    parser.add_argument('--dataroot', type=str, default="../data/", help='path to dataset')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--save_path', type=str, default=None, help='path to the model for possible saving')
    parser.add_argument('--load_path', type=str, default=None, help='path to the model for possible loading')
    args = parser.parse_args()

    return args

def main():
    """Main pipeline."""
    # parse command line arguments
    # load trainig set and split
    args = parse_args()
    transformations = transforms.Compose([
        transforms.Lambda(lambda samp: ((samp[0] / 127.5) - 1.0, samp[1])),
        Flip(0.3),
        ToTensor()]
        )
    full_dataset = CARLADataset(args.dataroot, transform=transformations)
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    print("==> Preparing dataset ...")

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4)
    # define model
    print("==> Initialize model ...")
    if args.load_path is not None:
        print("==> Loading model from supplied path ...")
        model = torch.load(args.load_path)
    else:
        model = NetworkNvidia().cuda()

    # define optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # learning rate scheduler
    scheduler = MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)

    # cuda or cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)

    # training
    print("==> Start training ...")
    train(model, criterion, optimizer, scheduler, train_loader, val_loader, args.save_path)


if __name__ == '__main__':
    main()