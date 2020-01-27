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
import argparse

def export_resnet50(dir_path):
    name = "resnet50"
    full_path = os.path.join(dir_path, name)
    onnx_model_path = os.path.join(full_path, name + ".onnx")
    torch_model_path = os.path.join(full_path, name + ".pt")
    os.makedirs(full_path, exist_ok=True)
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    input_size = [1, 3, 224, 224]
    x = torch.randn(input_size, requires_grad=True)
    # What about opset versions?
    torch.save(model, torch_model_path)
    torch.onnx.export(model, x, onnx_model_path, export_params=True, opset_version=11)
    save_metadata(model, name, full_path, input_size)

def export_fasterRCNN(dir_path):
    """
    export not working in pytorch 1.3? complains about FrozenBatchNorm

    uses resnet as backbone
    """
    name = "fasterrcnn"
    full_path = os.path.join(dir_path, name)
    onnx_model_path = os.path.join(full_path, name + ".onnx")
    torch_model_path = os.path.join(full_path, name + ".pt")
    os.makedirs(full_path, exist_ok=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    input_size = [1, 3, 224, 224]
    x = torch.randn(input_size, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.save(model, torch_model_path)
    torch.onnx.export(model, x, onnx_model_path, export_params=True, opset_version=11)
    save_metadata(model, name, full_path, input_size)

def export_squeezenet(dir_path):
    name = "squeezenet"
    full_path = os.path.join(dir_path, name)
    onnx_model_path = os.path.join(full_path, name + ".onnx")
    torch_model_path = os.path.join(full_path, name + ".pt")
    os.makedirs(full_path, exist_ok=True)
    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.eval()
    input_size = [1, 3, 224, 224]
    x = torch.randn(input_size, requires_grad=True)
    #out = model(x)
    # What about opset versions?
    torch.save(model, torch_model_path)
    # Tensorrt segmentation fault with opset version 11?
    torch.onnx.export(model, x, onnx_model_path, export_params=True, opset_version=9)
    save_metadata(model, name, full_path, input_size)

def export_lstm(dir_path):
    name = "lstm"
    full_path = os.path.join(dir_path, name)
    onnx_model_path = os.path.join(full_path, name + ".onnx")
    torch_model_path = os.path.join(full_path, name + ".pt")
    os.makedirs(full_path, exist_ok=True)
    layer_count = 4
    input_size = [5, 3, 10]

    model = nn.LSTM(10, 20, num_layers=layer_count, bidirectional=True)

    with torch.no_grad():
        input = torch.randn(input_size)
        h0 = torch.randn(layer_count * 2, 3, 20)
        c0 = torch.randn(layer_count * 2, 3, 20)
        output, (hn, cn) = model(input, (h0, c0))

        # default export
        #torch.onnx.export(model, (input, (h0, c0)), 'lstm.onnx')
        #onnx_model = onnx.load('lstm.onnx')
        # input shape [5, 3, 10]
        # print(onnx_model.graph.input[0])

        # export with `dynamic_axes`
        torch.save(model.state_dict, torch_model_path)
        torch.onnx.export(model, (input, (h0, c0)), onnx_model_path,
                        input_names=['input', 'h0', 'c0'],
                        output_names=['output', 'hn', 'cn'],
                        dynamic_axes={'input': {0: 'sequence'}, 'output': {0: 'sequence'}},
                        opset_version=11)
        # onnx_model = onnx.load('lstm.onnx')
        # input shape ['sequence', 3, 10]
        # print(onnx_model.graph.input[0])
    
    save_metadata(model, name, full_path, input_size)

def calculate_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_metadata(model, name, dir_path, input_size):
    import configparser
    config = configparser.ConfigParser()
    num_parameters = calculate_parameters(model)
    config['DEFAULT'] = {
        'model': name,
        'num_parameters': num_parameters,
        'input_size': input_size
        }

    with open(os.path.join(dir_path, 'metadata.cfg'), 'w') as configfile:
        config.write(configfile)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-path', type=str, help='Directory where models are saved.')
    parser = args.parse_args()
    #export_resnet50(parser.path)
    #export_lstm(parser.path)
    export_squeezenet(parser.path)
    #export_fasterRCNN(parser.path)