"""
Test models using tensorflow
"""

import numpy as np
from timeit import default_timer as timer

def run_tensorflow_inference(model, random_inputs, device):
    import tensorflow as tf
    #tf.compat.v1.disable_eager_execution()
    n = len(random_inputs)
    outputs = []
    random_inputs = np.expand_dims(random_inputs, 1)
    print("in function")
    #datas = tf.data.Dataset.from_tensor_slices(random_inputs).prefetch(1000)
    asd = model.predict_on_batch(random_inputs[0])
    print("got here at least")
    start = timer()
    #for data in datas:
    #    outputs.append(model.predict_on_batch(data))
    #model(data)
    for i in range(n):
        model.predict_on_batch(random_inputs[i])
    
    inference_time = (timer() - start)*1000 / n
    print(f'tensorflow inference time (msec): {inference_time:.5f}')
    return outputs, inference_time