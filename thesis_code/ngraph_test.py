import onnx
from ngraph_onnx.onnx_importer.importer import import_onnx_model
import ngraph as ng
import numpy as np
onnx_protobuf = onnx.load('models/resnet50/resnet50.onnx')
ng_function = import_onnx_model(onnx_protobuf)
print(ng_function)
runtime = ng.runtime(backend_name='CPU')
resnet_on_cpu = runtime.computation(ng_function)
import numpy as np
picture = np.ones([1, 3, 224, 224], dtype=np.float32)
picture = np.ones([1, 3, 224, 224], dtype=np.float32)
resnet_on_cpu(picture)