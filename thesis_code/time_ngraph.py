import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng
import numpy as np
from timeit import default_timer as timer

def run_ngraph_inference(model_onnx, inputs, device):
    onnx_protobuf = onnx.load(model_onnx)
    ng_function = import_onnx_model(onnx_protobuf)
    runtime = ng.runtime(backend_name=device)
    model = runtime.computation(ng_function)
    # TODO: doesnt work with fully connected, fix
    if "fully" not in model_onnx:
        inputs = np.expand_dims(inputs, 1) # add a dimension so slicing in the loop returns a properly shaped input for ngraph. pytorch too
    n = len(inputs)
    outputs = []

    start = timer()
    for i in range(n):
        data = inputs[i]
        outputs.append(model(data))
    
    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    return outputs, inference_time
