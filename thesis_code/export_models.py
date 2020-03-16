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
from utils import get_pytorch_model, get_tensorflow_model
# wonky sys thing required because yolo uses files in its own dir so this way the import works there
import sys
sys.path.append('model_definitions/')
from model_definitions import yolo, fully_connected, lstm

def export_model(model, dir_path, name, opset_version, input_size):
    full_path = os.path.join(dir_path, name)
    onnx_model_path = os.path.join(full_path, name + ".onnx")
    torch_model_path = os.path.join(full_path, name + ".pt")
    os.makedirs(full_path, exist_ok=True)
    x = torch.randn(input_size, requires_grad=True)

    model.eval()
    #torch.save(model, torch_model_path)
    torch.onnx.export(model, x, onnx_model_path, export_params=True, opset_version=opset_version, input_names=['input'], output_names=['output'])
    save_metadata(model, name, full_path, input_size, opset_version)

def lstm_prep():
    # Preps data and data sizes for lstm model
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
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

    # These will usually be more like 32 or 64 dimensional.
    # We will keep them small, so we can see how the weights change as we train.
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 32

    model = lstm.LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

    inputs = prepare_sequence(training_data[0][0], word_to_ix)

    return model, inputs
    
def calculate_parameters(model):
    return sum(p.numel() for p in model.parameters())

def save_metadata(model, name, dir_path, input_size, onnx_opset):
    # Writes some info about the model to a config file
    import configparser
    config = configparser.ConfigParser()
    num_parameters = calculate_parameters(model)
    config['DEFAULT'] = {
        'model': name,
        'num_parameters': num_parameters,
        'input_size': input_size,
        'onnx_opset': onnx_opset
        }

    with open(os.path.join(dir_path, 'metadata.cfg'), 'w') as configfile:
        config.write(configfile)
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-path', type=str, default="/l/dippa_main/dippa/thesis_code/models/", help='Directory where models are saved.')
    args.add_argument('-model', type=str, help='Name of the model, e.g. "resnet50"')
    args.add_argument('-opset', type=int, default=9, help='Which ONNX opset version to use')
    parser = args.parse_args()
    model_name = parser.model
    print("Initializing model {0}".format(model_name))
    model, input_size = get_pytorch_model(model_name)
    
    print("Exporting model {0}".format(model_name))
    export_model(model, parser.path, model_name, parser.opset, input_size)
