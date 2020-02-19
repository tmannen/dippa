"""
Test models using tensorflow
"""

import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

def run_tensorflow_inference(model, random_inputs, device):
    n = len(random_inputs)
    outputs = []
    start = timer()
    random_inputs = np.expand_dims(random_inputs, 1)
    #datas = tf.data.Dataset.from_tensor_slices(random_inputs)
    #for data in datas:
    #model(data)
    for i in range(n):
        outputs.append(model(random_inputs[i]))
    
    inference_time = (timer() - start)*1000 / n
    print(f'tensorflow inference time (msec): {inference_time:.5f}')
    return outputs, inference_time