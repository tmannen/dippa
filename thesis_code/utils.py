import pandas as pd
import numpy as np
import os
import csv
import seaborn as sns
import matplotlib as plt

from absl import app, flags, logging
from absl.flags import FLAGS


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')

def save_results(path, engine, model, time, n, device):
    # CSV with fields (engine, model, time, n)? appends to csv?
    save_path = os.path.join(path, "results.csv")
    fields = [engine, model, time, n, device]
    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def graph_results(results_path, exclude_models=None):
    # make a bar graph from results.csv? results has fields engine,model,time,n
    datas = pd.read_csv(results_path, header=None)
    datas.columns = ["method", "model", "time", "n", "device"] # Note: not needed if we add headers to csv
    datas['method'] = datas['method'] + " (" + datas['device'] + ")"
    sns.set(style="whitegrid")
    datas = datas[~datas.model.isin(exclude_models)]
    g = sns.catplot(x="model", y="time", hue="method", data=datas, kind="bar")
    g.set_ylabels("Average time per single inference (milliseconds)")
    plt.show()

def get_pytorch_model(name):
    import torchvision
    import sys
    import torch
    sys.path.append('model_definitions/')
    from model_definitions import yolo, fully_connected, lstm
    if name == "resnet50":
        input_size = [1, 3, 224, 224]
        model = torchvision.models.resnet50(pretrained=True)
    elif name == "mobilenet":
        input_size = [1, 3, 224, 224]
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif name == "squeezenet":
        input_size = [1, 3, 224, 224]
        model = torchvision.models.squeezenet1_0(pretrained=True)
    elif name == "fully_connected":
        input_size = [784]
        model = fully_connected.FullyConnected()
    elif name == "vgg16":
        input_size = [1, 3, 224, 224]
        model = torchvision.models.vgg16(pretrained=True)
    elif name == "yolo":
        input_size = [1, 3, 224, 224]
        model = yolo.Darknet("model_definitions/config/yolov3.cfg")
        model.load_darknet_weights("model_definitions/weights/yolov3.weights")
    elif name == "ssd":
        input_size = [1, 3, 300, 300]
        precision = 'fp32' # Maybe try other precisions? prolly not needed though?
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
    elif name == "lstm":
        # TODO: make this simpler. right now uses code in export_models in there and blaa
        pass

    return model, input_size

def get_tensorflow_model(name):
    import tensorflow as tf
    import sys
    sys.path.append('model_definitions/')
    from model_definitions import tensorflow_models
    from model_definitions.yolo_tf.models import YoloV3, YoloV3Tiny
    from model_definitions.yolo_tf.utils import load_darknet_weights
    if name == "resnet50":
        return tf.keras.applications.ResNet50()
    elif name == "mobilenet":
        return tf.keras.applications.MobileNetV2()
    elif name == "fully_connected":
        return tensorflow_models.get_fully_connected()
    elif name == "vgg16":
        return tf.keras.applications.VGG16()
    elif name == "yolo":
        # 80 seems to be the default, let's just go with that
        yolo = YoloV3(classes=80)
        load_darknet_weights(yolo, "weights/yolov3.weights", False)
        return yolo
    elif name == "lstm":
        # TODO: tf lstm
        pass

def get_tensorflow1_graph_from_onnx(name, strict=False):
    import tensorflow as tf
    import onnx
    from onnx_tf.backend import prepare

    # From: https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d
    def load_pb(path_to_pb):
        with tf.gfile.GFile(path_to_pb, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            return graph
    
    # First from onnx to pb tf. doing this way instead of onnx-tf graph shit because jostain vitun syystä
    # input ja output nimet ei toimi muuten??????? pitäiskö export_modelsissa tehdä tää?
    model_onnx = onnx.load("models/{0}/{0}.onnx".format(name))
    tf_rep = prepare(model_onnx, strict=strict)
    tf_rep.export_graph('models/{0}/{0}.pb'.format(name))
    tf_graph = load_pb("models/{0}/{0}.pb".format(name))
    return tf_graph