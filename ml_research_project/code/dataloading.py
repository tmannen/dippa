from torch.utils import data

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
        steering_angle = self.steer[index]
        rgb_img = np.array(Image.open(os.path.join(self.imagedir, str(index) + ".jpg")))

        return (rgb_img, steering_angle)

    def __len__(self):
        """Length of dataset."""
        return len(self.steer)