import tensorflow as tf

def load_pb(path_to_pb): 
    with tf.gfile.GFile(path_to_pb, 'rb') as f: 
        graph_def = tf.GraphDef() 
        graph_def.ParseFromString(f.read()) 
    with tf.Graph().as_default() as graph: 
        tf.import_graph_def(graph_def, name='') 
        return graph 

tf_graph = load_pb('resnet_from_onnx_to_tf.pb') 
sess = tf.Session(graph=tf_graph) 
output_tensor = tf_graph.get_tensor_by_name('495:0') 
input_tensor = tf_graph.get_tensor_by_name('input.1:0') 






from model_definitions import tensorflow_models
from model_definitions.yolo_tf.models import YoloV3, YoloV3Tiny
from model_definitions.yolo_tf.utils import load_darknet_weights
yolo = YoloV3(classes=80)
load_darknet_weights(yolo, "weights/yolov3.weights", False)