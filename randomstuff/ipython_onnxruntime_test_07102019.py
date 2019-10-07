 1/1: from numpy.random import binomial
 1/2: binomial(1, 0.3)
 1/3: binomial(1, 0.3)
 1/4: binomial(1, 0.3)
 1/5: binomial(1, 0.3)
 1/6: binomial(1, 0.3)
 1/7: binomial(1, 0.3)
 1/8: binomial(1, 0.3)
 1/9: binomial(1, 0.3)
1/10: binomial(1, 0.3)
1/11: binomial(1, 0.3)
1/12: from numpy.random import binomial
1/13: from numpy.random import binomial
1/14: binomial(1, 0.3)
1/15: binomial(1, 0.3)
1/16: binomial(1, 0.3)
1/17: binomial(1, 0.3)
1/18: binomial(1, 0.3)
1/19: binomial(1, 0.3)
1/20: binomial(1, 0.3)
1/21: binomial(1, 0.3)
1/22: binomial(1, 0.3)
1/23: binomial(1, 0.3)
 2/1: from numpy.random import binomial
 2/2: data = binomial(100, 0.3)
 2/3: print(data)
 2/4: data = binomial(1, 0.3, 1000)
 2/5: print(mean(data))
 2/6: import numpy as np
 2/7: print(np.mean(data))
 3/1: 2.79/23.3
 5/1: %paste
 5/2: %paste
 5/3: torch.cuda.is_available()
 6/1: import torch.onnx
 6/2: import torch
 6/3: import torchvision
 6/4: import numpy as np
 6/5: model = torchvision.models.resnet50(pretrained=True)
 6/6: ls
 6/7: ls
 6/8: from torch.autograd import Variable
 6/9: imagenet_input = Variable(torch.randn(1, 3, 224, 224))
6/10: torch.onnx.export(model, imagenet_input, 'resnet50.onnx')
6/11: ls
6/12:
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
6/13: import onnx
6/14:
onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
6/15: import onnxruntime
6/16: ort_session = onnxruntime.InferenceSession("resnet50.onnx")
6/17:
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
6/18: ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
   1: import torch.onnx
   2: import numpy as np
   3: import torch
   4: import torchvision
   5: model = torchvision.models.resnet50(pretrained=True)
   6: batch_size = 1
   7: model.eval()
   8: x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
   9: torch_out = model(x)
  10: x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
  11: torch_out = model(x)
  12: torch.onnx.export(model, x, "resnet50.onnx", export_params=True)
  13: %paste
  14:
import onnx

onnx_model = onnx.load("resnet50.onnx")
onnx.checker.check_model(onnx_model)
  15: import onnxruntime
  16: ort_session = onnxruntime.InferenceSession("resnet50.onnx")
  17:
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
  18: ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
  19: ort_outs = ort_session.run(None, ort_inputs)
  20: np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
  21: np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-09)
  22: ort_outs
  23: ort_outs.shape
  24: ort_outs[0].shape
  25: np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-15)
  26: np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-10, atol=1e-15)
  27: %history -g -f ipython_onnxruntime_test_07102019
  28: ls
  29: %history -g -f ipython_onnxruntime_test_07102019.py
