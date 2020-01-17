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

def get_and_save_paths(dir_path):
    name = dir_path.split("/")[-1]
    onnx_model_name = ".".join([name, "onnx"])
    torch_model_name = ".".join([name, "pt"])
    os.makedirs(dir_path, exist_ok=True)
    return name, onnx_model_name, torch_model_name

def export_resnet50(dir_path):
    name, onnx_model_name, torch_model_name = get_and_save_paths(dir_path)
    model = torchvision.models.resnet50(pretrained=True)
    model.eval()
    input_size = [1, 3, 224, 224]
    x = torch.randn(input_size, requires_grad=True)
    # What about opset versions?
    torch.save(model, os.path.join(dir_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(dir_path, onnx_model_name), export_params=True, opset_version=11)
    save_metadata(model, name, dir_path, input_size)

def export_fasterRCNN(dir_path):
    """
    export not working in pytorch 1.3? complains about FrozenBatchNorm

    uses resnet as backbone
    """
    name, onnx_model_name, torch_model_name = get_and_save_paths(dir_path)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.save(model, os.path.join(model_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(model_path, onnx_model_name), export_params=True, opset_version=11)

def export_squeezenet(dir_path):
    name, onnx_model_name, torch_model_name = get_and_save_paths(dir_path)
    model = torchvision.models.squeezenet1_0(pretrained=True)
    model.eval()
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = model(x)
    # What about opset versions?
    torch.save(model, os.path.join(dir_path, torch_model_name))
    torch.onnx.export(model, x, os.path.join(dir_path, onnx_model_name), export_params=True, opset_version=11)

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
    export_resnet50(parser.path)
    #export_lstm()
    export_squeezenet()