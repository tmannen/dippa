"""
Test models using pytorch

parts from: https://github.com/gthparch/edgeBench/blob/master/pytorch/execute.py
"""

import torch
import numpy as np
import argparse
import utils
from torch import cuda
from timeit import default_timer as timer

def run_pytorch_inference(pytorch_model_path, random_inputs, device='cpu'):
    model = torch.load(pytorch_model_path)
    model.eval()
    net = model.to(device)
    name = pytorch_model_path.split("/")[-1]
    n = len(random_inputs)
    outputs = []
    random_inputs = torch.from_numpy(np.expand_dims(random_inputs, 1)).to(device)

    start = timer()
    with torch.no_grad():
        for i in range(n):
            outputs.append(net(random_inputs[i]))
    
    inference_time = (timer() - start)*1000 / n
    print(f'{name} inference time (msec): {inference_time:.5f}')
    return outputs, inference_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_path', type=str, default=None, help='PyTorch Model path.') # Example: "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.onnx"
    parser.add_argument('-cpu', default=False, type=bool, help='Using CPU or not.')
    parser.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    parser.add_argument('-input_size', type=int, nargs='+', help='Input size (for ex. 3 224 224)', default=[3, 224, 224])
    args = parser.parse_args()
    n = args.n
    input_size = [n] + args.input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    outputs, inference_time = run_pytorch_inference(args.model_path, random_inputs)
    name = args.model_path.split("/")[-1].split(".")[0]
    utils.save_results("/l/dippa_main/dippa/thesis_code/models/", "pytorch", name, str(inference_time), args.n)
    # main()