import torch.onnx
import torch
import torchvision
import numpy as np
import onnx
import time
import onnxruntime


model_path = '../models/resnet50.onnx'

model = torchvision.models.resnet50(pretrained=True)
batch_size = 1
model.eval()
x = torch.randn(batch_size, 3, 224, 224, requires_grad=False)
torch_out = model(x)

start = time.time()
test_batch_size = 100
test_batch = torch.randn(test_batch_size, 3, 224, 224, requires_grad=False)
print(test_batch[0].shape)
#for i in range(100):
#    model(test_batch[None, i])

end = time.time()
print("Time taken on average for one image with pytorch eval: ", (end-start)/test_batch_size)

onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)
ort_session = onnxruntime.InferenceSession(model_path)
ort_inputs = {ort_session.get_inputs()[0].name: x}
ort_outs = ort_session.run(None, ort_inputs)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.numpy()

start = time.time()
for i in range(100):
    ort_session.run(None, {input_name: to_numpy(test_batch[None, i])})

end = time.time()
print("Time taken on average for one image with onnxruntime: ", (end-start)/test_batch_size)