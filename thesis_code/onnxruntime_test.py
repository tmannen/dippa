import numpy as np
import onnxruntime as ot
from timeit import default_timer as timer

sess_options = ot.SessionOptions()

sess_options.graph_optimization_level = ot.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.optimized_model_filepath = "optimized_network.onnx"
sess = ot.InferenceSession("models/resnet50/resnet50.onnx", sess_options=sess_options)
n = 1000
rand_x = np.random.randn(1, 3, 224, 224).astype(np.float32)

outputs = sess.run(None, {'inputs': rand_x})

del sess

sess = ot.InferenceSession("optimized_network.onnx")

outputs = sess.run(None, {'inputs': rand_x})

start = timer()
rand_many = np.random.randn(n, 3, 224, 224).astype(np.float32)

for i in range(n):
    outputs = sess.run(None, {'model_input': rand_many[i]})

inference_time = (timer() - start)*1000 / n     
print(f'inference time (msec): {inference_time:.5f}')
