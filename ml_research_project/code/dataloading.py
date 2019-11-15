from torch.utils import data
import torch
from torchvision import transforms

#import cv2
import numpy as np
import os
from PIL import Image

class CARLADataset(data.Dataset):
    """
    Gets images, steering angles from directory

    Note: maybe it would be better to load straight from h5py? But how best to load multiple h5 files dynamically?
    """

    def __init__(self, dataroot, transform=None):
        """Initialization."""
        self.transform = transform
        self.dataroot = dataroot
        self.imagedir = os.path.join(self.dataroot, "rgb")
        self.steer = np.load(os.path.join(self.dataroot, "steering_angles.npy")).flatten()

    def __getitem__(self, index):
        """Get image."""
        steering_angle = np.array([self.steer[index]])
        rgb_img = np.array(Image.open(os.path.join(self.imagedir, str(index) + ".jpg")))
        rgb_img, steering_angle = self.transform((rgb_img, steering_angle))

        return rgb_img, steering_angle

    def __len__(self):
        """Length of dataset."""
        return len(self.steer)

class Flip(object):
    """
    Flip image and steer angle
    """

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        rgb_img, angle = sample
        if np.random.rand() < self.flip_prob:
            rgb_img = np.fliplr(rgb_img)
            angle = angle * -1.0

        return rgb_img, angle

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb_img, angle = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb_img = rgb_img.transpose((2, 0, 1))
        return torch.from_numpy(rgb_img.copy()), torch.from_numpy(angle)

class ToTensorImg(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgb_img = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        rgb_img = rgb_img.transpose((2, 0, 1))
        return torch.from_numpy(rgb_img.copy())