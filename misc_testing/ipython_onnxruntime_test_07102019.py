import torch.onnx
import torch
import torchvision
import numpy as np
import onnx
from torch.autograd import Variable

model = torchvision.models.resnet50(pretrained=True)
batch_size = 1
model.eval()
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
model.eval()
torch_out = model(x)

# What about opset versions?
torch.onnx.export(model, x, 'resnet50.onnx', export_params=True)

onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)

import onnxruntime
ort_session = onnxruntime.InferenceSession("resnet50.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
