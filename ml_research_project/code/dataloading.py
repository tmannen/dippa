"""
Self-driving car image pair Dataset.
@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

from torch.utils import data

import cv2
import numpy as np

class CARLADataset(data.Dataset):
    def __init__(self, dataroot, samples, transform=None):
        """Initialization."""
        self.samples = samples
        self.transform = transform
        data = h5py.File("with_noise_1/data1.hdf5", 'r')
        sensors = data['sensors']
        steer = data['state/steer']
        metadata = data['metadata']
        metadata = json.loads(metadata[()])

        # Find correct index for ego_vehicle
        ego_id = metadata["ego_vehicle_id"]
        ego_index = None
        for index, actor_id in enumerate(data['state/id']):
            if actor_id[0, 0] == ego_id:
                ego_index = index

    def __getitem__(self, index):
        """Get image."""
        batch_samples  = self.samples[index]
        steering_angle = float(batch_samples[3])

        center_img, steering_angle_center = augment(self.dataroot, batch_samples[0], steering_angle)
        left_img, steering_angle_left     = augment(self.dataroot, batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right   = augment(self.dataroot, batch_samples[2], steering_angle - 0.4)

        center_img = self.transform(center_img)
        left_img   = self.transform(left_img)
        right_img  = self.transform(right_img)

        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        """Length of dataset."""
        return len(self.samples)