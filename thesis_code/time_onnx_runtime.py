import onnxruntime as rt
import numpy as np
from timeit import default_timer as timer

def run_onnx_runtime_inference(model_onnx, inputs, device):
    sess_options = rt.SessionOptions()

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    # TODO: doesnt work with fully connected, fix
    inputs = np.expand_dims(inputs, 1) # add a dimension so slicing in the loop returns a properly shaped input
    n = len(inputs)
    outputs = []
    times = []

    sess = rt.InferenceSession(model_onnx, sess_options=sess_options)
    for j in range(n//10):
        sess.run(['output'], {'input': inputs[np.random.randint(0, n-1)]})
        
    start = timer()
    for i in range(n):
        prev = timer()
        outputs.append(sess.run(['output'], {'input': inputs[i]}))
        times.append(timer() - prev)
    
    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    return outputs, times, inference_time

#x = np.random.random((1000,3,300,300)).astype(np.float32)
#run_onnx_runtime_inference("models/ssd/ssd.onnx", x, "cpu")