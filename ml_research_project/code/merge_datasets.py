"""
Create a single folder from many datasets. The datasets come in folders of hdf5 files, and these have their own indexes.
Make it so that all these extracted folders (made with create_dataset_new.py) are combined into a single folder.
Makes it easier to load data? Or maybe the dataloader should handle this?

TODO: handle different npy formats in new and old. global index. get all folders within some folder and then from there?
"""

import h5py
from PIL import Image
import json
import os
import numpy as np
import argparse
import sys
from shutil import copyfile

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser()
    # directory
    parser.add_argument('--source', type=str, help='path to where the files are')
    parser.add_argument('--target', type=str, help='path to where they need to be extracted')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    sourcedir = args.source
    # Create folder with the same name as source
    target_full = args.target
    if not os.path.exists(target_full):
        os.makedirs(target_full)
    datas = sorted(os.listdir(sourcedir))
    datas = [d for d in datas if d != "merged"]
    print(datas)
    imagedir = "rgb"
    target_imagedir = os.path.join(target_full, imagedir)
    if not os.path.exists(target_imagedir):
        os.makedirs(target_imagedir)
    global_index = 0
    steering_angles = []

    for folder in datas:
        steering_angles.append(np.load(os.path.join(sourcedir, folder, "steering_angles.npy")).flatten())
        source_imagedir = os.path.join(sourcedir, folder, imagedir)
        # Get sensor data and steering angle for all frames
        for image in sorted(os.listdir(source_imagedir), key=lambda x: int(x.split(".")[0])):
            copyfile(os.path.join(source_imagedir, image), os.path.join(target_imagedir, str(global_index) + "." + image.split(".")[-1]))
            global_index += 1

    all_steers = np.concatenate(steering_angles)
    np.save(os.path.join(target_full, "steering_angles"), all_steers)

if __name__ == '__main__':
    main()