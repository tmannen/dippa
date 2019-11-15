"""
Creates rgb, steer angle, more? from all the h5py files and puts them in the same folder
Needed? for smoother pytorch dataloading.
"""

import h5py
from PIL import Image
import json
import os
import numpy as np
import argparse
import sys

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()
    # directory
    parser.add_argument('--source', type=str, help='path to place where hdf5 files are')
    parser.add_argument('--target', type=str, help='path to where they need to be extracted')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    orig_data_root = args.source
    dataset_root = args.target

    datas = [h5py.File(os.path.join(orig_data_root, hdf), 'r') for hdf in os.listdir(orig_data_root) if ".hdf5" in hdf]
    steering_angles = []
    global_index = 0

    for data in datas:
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

        steering_angles.append(np.array(steer[ego_index]))
        # Get sensor data and steering angle for all frames
        for name, images in sensors.items():
            if "lidar" in name:
                # Not implemented
                continue
            # Make folder
            directory = os.path.join(dataset_root, name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            # Save images
            for index, data in enumerate(images):
                im = Image.fromarray(data)
                im.save(os.path.join(directory, "{}.jpg".format(global_index + index)))
        
        # Index has the last index of the images, add to global index at the end so all images are in order
        # +1 because index always starts at 0
        global_index += index + 1
        print("yee")

    all_steers = np.concatenate(steering_angles)
    np.save(dataset_root + "steering_angles", all_steers)

if __name__ == '__main__':
    main()