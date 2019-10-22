"""
Testing pytorch native inference vs. onnx runtime inference. 
"""

import torch.onnx
import torch
import torchvision
import numpy as np
import onnx
import time
import onnxruntime
import argparse

print(onnxruntime.get_device())
parser = argparse.ArgumentParser()
parser.add_argument('-gpu', default="True", type=str)
args = parser.parse_args()

model_path = 'models/resnet50.onnx'

if args.gpu=="True":
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = torchvision.models.resnet50(pretrained=True)
else:
    # is this enough?
    torch.set_default_tensor_type(torch.FloatTensor)
    model = torchvision.models.resnet50(pretrained=True)

batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=False)
model.eval()
torch_out = model(x)


start = time.time()
test_batch_size = 100
test_batch = torch.randn(test_batch_size, 3, 224, 224, requires_grad=False)

for i in range(100):
    model(test_batch[None, i])

end = time.time()
print("Time taken on average for one image with pytorch eval: ", (end-start)/test_batch_size)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(model_path)
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

test_batch_cpu = test_batch.cpu().numpy()
start = time.time()
for i in range(100):
    sample = test_batch_cpu[None, i]
    ort_session.run(None, {ort_session.get_inputs()[0].name: sample})

end = time.time()
print("Time taken on average for one image with onnxruntime: ", (end-start)/test_batch_size)