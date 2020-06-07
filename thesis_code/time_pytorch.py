"""
Test models using pytorch

parts from: https://github.com/gthparch/edgeBench/blob/master/pytorch/execute.py
"""

import torch
import numpy as np
import argparse
import utils
#from torch2trt import torch2trt
from torch import cuda
from timeit import default_timer as timer

def pytorch_single_inf(model, input, device):
    with torch.no_grad():
        output = model(torch.from_numpy(input).to(device))
        torch.cuda.synchronize()
    return output

def run_pytorch_inference(model, random_inputs, device='cuda'):
    device = "cuda" if device == "gpu" else device
    model = model.to(device)
    model.eval()
    n = len(random_inputs)
    outputs = []
    random_inputs = np.expand_dims(random_inputs, 1)
    #random_inputs = torch.from_numpy(np.expand_dims(random_inputs, 1)).to(device)
    #sampledata = torch.ones((1, 3, 224, 224)).cuda()
    #net = torch2trt(model, [sampledata])
    outputs = []
    times = []
    # warmup:
    #for j in range(n//10):
    #    model(torch.from_numpy(random_inputs[np.random.randint(0, n-1)]).to(device))
    start = timer()
    # If statement so we dont have extra if statements within loop. also copy back to cpu only if gpu is used
    with torch.no_grad():
        for i in range(n):
            prev = timer()
            if device == "cpu":
                outputs.append(model(torch.from_numpy(random_inputs[i])).numpy())
            else:
                outputs.append(model(torch.from_numpy(random_inputs[i]).to(device)).to("cpu").numpy())
                torch.cuda.synchronize()
            times.append(timer() - prev)
    
    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    return outputs, times, inference_time

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
