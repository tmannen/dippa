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
    #datas = tf.data.Dataset.from_tensor_slices(random_inputs).prefetch(1000)
    asd = model.predict_on_batch(random_inputs[0])
    start = timer()
    #for data in datas:
    #    outputs.append(model.predict_on_batch(data))
    #model(data)
    for i in range(n):
        model.predict_on_batch(random_inputs[i])
    
    inference_time = (timer() - start)*1000 / n
    print(f'tensorflow inference time (msec): {inference_time:.5f}')
    return outputs, inference_time

def run_tensorflow1_inference(model, random_inputs, device):
    import tensorflow as tf
    #sess = tf.Session(graph=model)
    n = len(random_inputs)
    random_inputs = np.expand_dims(random_inputs, 1)
    outputs = np.zeros((n, 1000))
    times = np.zeros(n)
    outputs = []
    times = []
    #sess.run(output_tensor, feed_dict={input_tensor: random_inputs[0]})
    with tf.Session(graph=model) as sess:
        output_tensor = model.get_tensor_by_name('output:0')
        input_tensor = model.get_tensor_by_name('input:0')
        # warmup
        for j in range(n//10):
            sess.run(output_tensor, feed_dict={input_tensor: random_inputs[np.random.randint(0, n-1)]})

        start = timer() 
        for i in range(n):
            prev = timer()
            outputs.append(sess.run(output_tensor, feed_dict={input_tensor: random_inputs[i]}))
            #outputs[i] = sess.run(output_tensor, feed_dict={input_tensor: random_inputs[i]})
            #times[i] = timer() - prev
            times.append(timer() - prev)
    
    inference_time = (timer() - start)*1000 / n
    print(f'tensorflow inference time (msec): {inference_time:.5f}')
    return outputs, times, inference_time

"""
import utils
model = utils.get_tensorflow1_graph_from_onnx("resnet50")
from time_tensorflow import run_tensorflow1_inference
run_inference = run_tensorflow1_inference
import numpy as np
device = "gpu"
inputs = np.random.randn(100, 3, 224, 224).astype(np.float32)
outputs, times, inference_time = run_inference(model, inputs, device)
"""