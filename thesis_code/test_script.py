"""
test different backends, most of them have python bindings etc. see later if C++ needed.

Do it like this?: command line args, different keywords and then each tool in its own function.

Do a timing function that just takes in the predictor (loading and other stuff is done before the function is called), 
and do inference n times with random numbers. Maybe another function that tests accuracy?
"""

import torch.onnx
import torch
import torchvision
import numpy as np
import onnx
from timeit import default_timer as timer
import onnxruntime
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-method', type=str)
args = parser.parse_args()

def time_inference(model):
