import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carlaegg/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
from carla import ColorConverter as cc
import math
import numpy as np
import torch
#from torchvision import transforms
#from dataloading import CARLADataset, Flip, ToTensorImg


def load_pytorch_model():
    model = torch.load("../models/testmodel.pth")

    return model

def transform_image(img):
    # same transforms as in training, also swap dims for torch
    img = (img / 127.5) - 1.0
    img = img.transpose((2, 0, 1))
    img = np.expand_dims(img, 0)
    return torch.from_numpy(img).float().cuda()
    
class example():

    def __init__(self):
        self.ego_vehicle_blueprint_keyword = "vehicle.audi.tt"
        self.ego_vehicle_color = "66,77,90"  # RGB
        self.resolution = (512, 256)
        self.control = carla.VehicleControl()
        print("Loading model..")
        self.model = load_pytorch_model()

    def parse_image(self, image):
        # https://carla.readthedocs.io/en/latest/python_api/#carlaimagecarlasensordata-class
        # Transform image to the same format as in the HDF5 files
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        # ::-1 reverses list
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
        torch_arr = transform_image(array)
        self.control.throttle = 0.3
        steer_angle = self.model(torch_arr).data.cpu().numpy().item()
        self.control.steer = steer_angle

    def main(self):
        # Connect to CARLA server
        self.client = carla.Client("127.0.0.1", 2000, worker_threads=0)  # worker_threads=0 means unlimited
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()

        self.spectator = self.world.get_spectator()

        # Set weather
        weather = carla.WeatherParameters(
            cloudyness=10.0,
            precipitation=0.0,
            sun_altitude_angle=70.0)
        weather.precipitation_deposits = 0
        self.world.set_weather(weather)

        # Create ego vehicle and sensors
        self.create_vehicle()
        self.create_sensors()

        # Start listening to CARLA ticks (one tick per frame)
        self.world.on_tick(lambda world_snapshot: self.tick(world_snapshot))

        while True:
            # Everything happens in self.tick and self.parse_image
            pass

    def tick(self, world_snapshot):
        # This runs on every frame
        self.ego_vehicle.apply_control(self.control)
        self.set_spectator_follow_transform()

    def create_vehicle(self):
        # Spawn ego vehicle
        blueprints = self.world.get_blueprint_library()
        ego_vehicle_blueprint = blueprints.filter(self.ego_vehicle_blueprint_keyword)
        blueprint = ego_vehicle_blueprint[0]

        # Set ego vehicle color (this doesnt really matter)
        if blueprint.has_attribute('color'):
            blueprint.set_attribute('color', self.ego_vehicle_color)

        # Get list of spawn points
        spawn_points = self.world.get_map().get_spawn_points()

        # Spawn vehicle on the first spawn point
        self.ego_vehicle = self.world.spawn_actor(blueprint, spawn_points[0])

    def set_spectator_follow_transform(self):
        # Follow ego vehicle in the simulation window
        ego_vehicle_transform = self.ego_vehicle.get_transform()
        spectator_location = ego_vehicle_transform.location
        spectator_rotation = ego_vehicle_transform.rotation

        # Snap camera rotation to prevent nausea
        angle = -180
        snapped = False
        while angle < 180:
            if spectator_rotation.yaw > angle - 45 / 2 and spectator_rotation.yaw < angle + 45 / 2:
                spectator_rotation.yaw = angle
                snapped = True
                break
            angle += 45
        if not snapped:
            spectator_rotation.yaw = 180
        a = math.radians(spectator_rotation.yaw)
        x = -6 * math.cos(a)
        y = -6 * math.sin(a)
        spectator_location_adjusted = carla.Location(spectator_location.x + x, spectator_location.y + y,
                                                     spectator_location.z + 2.5)
        spectator_rotation_adjusted = carla.Rotation(0, spectator_rotation.yaw, 0)
        spectator_transform = carla.Transform(spectator_location_adjusted, spectator_rotation_adjusted)
        self.spectator.set_transform(spectator_transform)

    def create_sensors(self):
        # Spawn camera
        bp_library = self.world.get_blueprint_library()
        bp = bp_library.find('sensor.camera.rgb')

        bp.set_attribute('image_size_x', str(self.resolution[0]))
        bp.set_attribute('image_size_y', str(self.resolution[1]))

        self.sensor = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.6, z=1.7)),
            attach_to=self.ego_vehicle,
            attachment_type=carla.AttachmentType.Rigid)

        self.sensor.listen(lambda data: self.parse_image(data))

if __name__ == '__main__':
    e = example()
    e.main()
