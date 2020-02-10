import torch
import torchvision
import tensorrt as trt
import trt_common as common

TRT_LOGGER = trt.Logger()

input_size = [1, 3, 224, 224]
x = torch.randn(input_size, requires_grad=True)
model_export = torchvision.models.squeezenet1_0(pretrained=True)
torch.onnx.export(model_export, x, "squeezenet.onnx", export_params=True, opset_version=11)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
    with open("squeezenet.onnx", 'rb') as model:
        builder.max_workspace_size = 1 << 30 # 256MiB
        builder.max_batch_size = 1
        parser.parse(model.read())