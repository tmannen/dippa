import h5py
from PIL import Image
import json
import os

data = h5py.File("/l/raw_data/extracted/2019-11-15-set2/data20.hdf5", 'r')
sensors = data['sensors']
steer = data['state/steer']
metadata = data['metadata']
metadata = json.loads(metadata[()])

# Find correct index for ego_vehicle
ego_id = metadata["ego_vehicle_id"]
ego_index = None
for index, actor_id in enumerate(data['state/id'][0]):
    if actor_id == ego_id:
        ego_index = index

# Get sensor data and steering angle for all frames
for name, images in sensors.items():
    if "lidar" in name:
        # Not implemented
        continue
    # Make folder
    directory = "_test/" + name + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save images
    for index, data in enumerate(images):
        im = Image.fromarray(data)
        im.save(directory + "{}.jpg".format(index))
        print(steer[index, ego_index])

    print(index)