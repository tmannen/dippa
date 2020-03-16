import tensorflow as tf
from tensorflow.keras import layers

def get_resnet50():
    return tf.keras.applications.ResNet50()

def get_mobilenetv2():
    return tf.keras.applications.MobileNetV2()

def get_fully_connected():
    model = tf.keras.Sequential([
        layers.Dense(100, input_shape=(784,)),
        layers.Activation('relu'),
        layers.Dropout(rate=0.5),
        layers.Dense(100),
        layers.Activation('relu'),
        layers.Dense(10), asd
    ])

    return model

### TODO: get yolo from here: https://github.com/zzh8829/yolov3-tf2
### TODO: maybe lstm same as pt?