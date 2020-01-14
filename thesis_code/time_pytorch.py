"""
Test models using pytorch

parts from: https://github.com/gthparch/edgeBench/blob/master/pytorch/execute.py
"""

import torch
from torch import cuda
from timeit import default_timer as timer

def run_pytorch_inference(pytorch_model_path, n=1000):
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = torch.load(pytorch_model_path)
    model.eval()
    net = model.to(device)
    name = pytorch_model_path.split("/")[-1]
    outputs = []

    start = timer()
    with torch.no_grad():
        for random_input in random_inputs:
            data = torch.randn(1, 3, 224, 224).to(device)
            outputs.append(net(data))
    print(f'{name} inference time (msec): {(timer() - start)*1000 / n:.5f}')
    return outputs

if __name__ == '__main__':
    run_pytorch_inference("/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.pt")
    # main()