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
import models

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

    ### stuff to get input nicely, from: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html?
    def prepare_sequence(seq, to_ix):
        idxs = [to_ix[w] for w in seq]
        return torch.tensor(idxs, dtype=torch.long)

    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    model = models.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

    inputs = prepare_sequence(training_data[0][0], word_to_ix)

    with torch.no_grad():
        # export with `dynamic_axes`
        torch.save(model, torch_model_path)
        torch.onnx.export(model, inputs, onnx_model_path, opset_version=9)
    
    save_metadata(model, name, full_path, "Variable")

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