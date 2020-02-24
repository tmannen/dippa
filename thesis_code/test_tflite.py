import tensorflow as tf
import numpy as np
from timeit import default_timer as timer

model = tf.keras.applications.ResNet50()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
n = 1000
input_shape = input_details[0]['shape']
random_inputs = np.array(np.random.random_sample((n, 1, 224, 224, 3)), dtype=np.float32)
outputs = []
interpreter.set_tensor(input_details[0]['index'], random_inputs[0])
interpreter.invoke()
start = timer()
interpreter.set_tensor(input_details[0]['index'], random_inputs[0])
for i in range(n):
    #interpreter.set_tensor(input_details[0]['index'], random_inputs[i])
    interpreter.invoke()
    #tflite_results = interpreter.get_tensor(output_details[0]['index'])

inference_time = (timer() - start)*1000 / n
print(f'tensorflow inference time (msec): {inference_time:.5f}')