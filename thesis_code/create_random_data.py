"""
for onnx perf test
"""

from onnx import numpy_helper
import numpy as np
import onnx

numpy_array = np.random.randn(1, 3, 224, 224).astype(np.float32)
tensor = numpy_helper.from_array(numpy_array)
with open('/l/dippa_main/dippa/thesis_code/models/resnet50/test_data_set_0/input_0.pb', 'wb') as f:
    f.write(tensor.SerializeToString())