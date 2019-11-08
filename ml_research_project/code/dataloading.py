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
        steering_angle = torch.from_numpy(np.array(self.steer[index]))
        rgb_img = Image.open(os.path.join(self.imagedir, str(index) + ".jpg")).transpose((2, 0, 1))
        rgb_img, steering_angle = self.transform(rgb_img)

        return rgb_img, steering_angle

    def __len__(self):
        """Length of dataset."""
        return len(self.steer)

class Flip(object):
    """
    Flip image and steer angle
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        rgb_img, angle = sample

        if np.random.rand() < 0.3:
            rgb_img = transforms.RandomHorizontalFlip(1.0) # function doesn't say whether image was flipped or not so we do it this way
            angle = angle * -1.0

        return rgb_img, angle