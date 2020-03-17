import onnxruntime as rt
import numpy as np
from timeit import default_timer as timer

def run_onnx_runtime_inference(model_onnx, inputs, device):
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # TODO: doesnt work with fully connected, fix
    inputs = np.expand_dims(inputs, 1) # add a dimension so slicing in the loop returns a properly shaped input for ngraph. pytorch too
    n = len(inputs)
    outputs = []

    sess = rt.InferenceSession(model_onnx, sess_options=sess_options)
    start = timer()
    for i in range(n):
        data = inputs[i]
        sess.run(['output'], {'input': data})
    
    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    return outputs, inference_time

x = np.random.random((1000,3,300,300)).astype(np.float32)
run_onnx_runtime_inference("models/ssd/ssd.onnx", x, "cpu")