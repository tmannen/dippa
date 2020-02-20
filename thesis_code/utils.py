import pandas as pd
import numpy as np
import os
import csv
import torch
import seaborn as sns
import matplotlib as plt

def save_results(path, engine, model, time, n, device):
    # CSV with fields (engine, model, time, n)? appends to csv?
    save_path = os.path.join(path, "results.csv")
    fields = [engine, model, time, n, device]
    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def graph_results(results_path):
    # make a bar graph from results.csv? results has fields engine,model,time,n
    datas = pd.read_csv(results_path, header=None)
    datas.columns = ["method", "model", "time", "n", "device"] # Note: not needed if we add headers to csv
    datas['method'] = datas['method'] + " (" + datas['device'] + ")"
    sns.set(style="whitegrid")
    g = sns.catplot(x="model", y="time", hue="method", data=datas, kind="bar")
    g.set_ylabels("Average time per single inference (milliseconds)")
    plt.show()

def get_pytorch_model(name):
    import torchvision
    import sys
    sys.path.append('model_definitions/')
    from model_definitions import yolo, fully_connected, lstm
    if name == "resnet50":
        return torchvision.models.resnet50(pretrained=True)
    elif name == "mobilenet":
        return torchvision.models.mobilenet_v2(pretrained=True)
    elif name == "squeezenet":
        torchvision.models.squeezenet1_0(pretrained=True)
    elif name == "fully_connected":
        return fully_connected.FullyConnected()
    elif name == "yolo":
        model = yolo.Darknet("model_definitions/config/yolov3.cfg")
        model.load_darknet_weights("model_definitions/weights/yolov3.weights")
        return model
    elif name == "lstm":
        # TODO: make this simpler. right now uses code in export_models in there and blaa
        pass

def get_tensorflow_model(name):
    import tensorflow as tf
    import sys
    sys.path.append('model_definitions/')
    from model_definitions.yolo_tf.models import YoloV3, YoloV3Tiny
    from model_definitions.yolo_tf.utils import load_darknet_weights
    if name == "resnet50":
        return tf.keras.applications.ResNet50()
    elif name == "mobilenet":
        return tf.keras.applications.MobileNetV2()
    elif name == "fully_connected":
        return tensorflow_models.get_fully_connected()
    elif name == "yolo":
        # 80 seems to be the default, let's just go with that
        yolo = YoloV3(classes=80)
        load_darknet_weights(yolo, "weights/yolov2.weights", False)
        return yolo
    elif name == "lstm":
        # TODO: tf lstm
        pass