"""
export several models from pytorch to ONNX. These are used later from other inference engines.
Also export the original pytorch model.
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
import os

def export_resnet50():
    # TODO: absolute paths?
    model_path = "models/resnet50/"
    os.makedirs(model_path, exist_ok=True)
    onnx_model_name = "resnet50.onnx"
    torch_model_name = "resnet50.pt"
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    # What about opset versions?
    torch.save(model, os.path.join(model_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(model_path, onnx_model_name), export_params=True, opset_version=11)

def export_fasterRCNN():
    """
    export not working in pytorch 1.3? complains about FrozenBatchNorm

    uses resnet as backbone
    """
    model_path = "models/faster_rcnn/"
    os.makedirs(model_path, exist_ok=True)
    onnx_model_name = "faster_rcnn.onnx"
    torch_model_name = "faster_rcnn.pt"
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.save(model.state_dict, os.path.join(model_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(model_path, onnx_model_name), export_params=True, opset_version=11)

def export_squeezenet():
    model_path = "models/squeezenet/"
    os.makedirs(model_path)
    onnx_model_name = "squeezenet.onnx"
    torch_model_name = "squeezenet.pt"
    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.save(model.state_dict, os.path.join(model_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(model_path, onnx_model_name), export_params=True, opset_version=11)

def export_lstm():
    model_path = "models/lstm/"
    os.makedirs(model_path)
    onnx_model_name = "lstm.onnx"
    torch_model_name = "lstm.pt"
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
        torch.save(model.state_dict, os.path.join(model_path, torch_model_name))
        torch.onnx.export(model, (input, (h0, c0)), 'models/lstm.onnx',
                        input_names=['input', 'h0', 'c0'],
                        output_names=['output', 'hn', 'cn'],
                        dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}},
                        opset_version=11)
        # onnx_model = onnx.load('lstm.onnx')
        # input shape ['sequence', 3, 10]
        # print(onnx_model.graph.input[0])

"""
def export_yolo():
    import models
    yolo = models.Darknet('config/yolov3.cfg')
    data_shape = [1, 3, 224, 224]
    import torch
    input = torch.randn(data_shape)
    yolo(input)
    import torch.onnx
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    torch.onnx.export(yolo, x, "models/yolo.onnx", export_params=True, opset_version=11)
    pwd
    torch.onnx.export(yolo, x, "yolo.onnx", export_params=True, opset_version=11)
    history
"""

export_resnet50()
#export_lstm()
#export_squeezenet()