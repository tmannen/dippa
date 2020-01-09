"""
export several models from pytorch to ONNX. These are used later from other inference engines.
"""

import torch.onnx
import torch
import torchvision
import numpy as np
import onnx
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
import torch.optim as optim

def export_resnet50():
    model_path = "models/resnet50/resnet50.onnx"
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    # What about opset versions?
    torch.onnx.export(model, x, model_path, export_params=True, opset_version=11)

def export_fasterRCNN():
    """
    not working in pytorch 1.3? complains about FrozenBatchNorm
    """
    model_path = "models/faster_rcnn.onnx"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.onnx.export(model, x, model_path, export_params=True, opset_version=11)

def export_squeezenet():
    """
    not working in pytorch 1.3? complains about FrozenBatchNorm
    """
    model_path = "models/squeezenet.onnx"
    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.onnx.export(model, x, model_path, export_params=True, opset_version=11)

def export_lstm():
    layer_count = 4

    model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)

    with torch.no_grad():
        input = torch.randn(5, 3, 10)
        h0 = torch.randn(layer_count * 2, 3, 20)
        c0 = torch.randn(layer_count * 2, 3, 20)
        output, (hn, cn) = model(input, (h0, c0))

        # default export
        #torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
        #onnx_model = onnx.load('lstm.onnx')
        # input shape [5, 3, 10]
        # print(onnx_model.graph.input[0])

        # export with `dynamic_axes`
        torch.onnx.export(model, (input, (h0, c0)), 'models/lstm.onnx',
                        input_names=['input', 'h0', 'c0'],
                        output_names=['output', 'hn', 'cn'],
                        dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}},
                        opset_version=11)
        # onnx_model = onnx.load('lstm.onnx')
        # input shape ['sequence', 3, 10]
        # print(onnx_model.graph.input[0])
def export_yolo():
    ls
    import models
    yolo = models.Darknet('config/yolo3.cfg')
    ls
    ls config
    yolo = models.Darknet('config/yolov3.cfg')
    data_shape = [1, 3, 224, 224]
    import torch
    input = torch.(data_shape)
    input = torch.randn(data_shape)
    input
    yolo(input)
    import torch.onnx
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(yolo, x, "models/yolo.onnx", export_params=True, opset_version=11)
    pwd
    torch.onnx.export(yolo, x, "yolo.onnx", export_params=True, opset_version=11)
    history


export_squeezenet()