import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng
import numpy as np
from timeit import default_timer as timer

def run_ngraph_inference(model_onnx, inputs, device):
    onnx_protobuf = onnx.load(model_onnx)
    ng_function = import_onnx_model(onnx_protobuf)
    # no GPU support in this
    runtime = ng.runtime(backend_name="cpu")
    model = runtime.computation(ng_function)
    # TODO: doesnt work with fully connected, fix
    inputs = np.expand_dims(inputs, 1) # add a dimension so slicing in the loop returns a properly shaped input for ngraph. pytorch too
    n = len(inputs)
    outputs = []
    times = []

    for j in range(n//10):
        model(inputs[np.random.randint(0, n-1)])

    start = timer()
    for i in range(n):
        prev = timer()
        outputs.append(model(inputs[i]))
        times.append(timer() - prev)
    
    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    return outputs, times, inference_time
