"""
Create a single file for timing models. Import different inferences 
from other places (like run_tensorrt_inference) and just control what tool to use with an if statement here
"""

import argparse
import numpy as np
import os
import utils
#from time_tensorrt import run_tensorrt_inference
#from time_pytorch import run_pytorch_inference
import pdb
## This is for the YOLO model:
import sys
sys.path.append('model_definitions/')

model_root_path = "/l/dippa_main/dippa/thesis_code/models/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default=None, help='Model name, for example "resnet50".')
    parser.add_argument('-method', type=str, help='pytorch, tensorrt, ngraph, tensorflow or openvino')
    parser.add_argument('-device', type=str, default="gpu", help='Device to run the model on, usually cpu or gpu')
    parser.add_argument('-save', default=False, type=bool, help='Whether to save the results to a CSV or not')
    parser.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    parser.add_argument('-input_size', type=int, nargs='+', help='Input size (for ex. 3 224 224)', default=[3, 224, 224])
    parser.add_argument('-simplified', type=bool, default=False, help='Use simplified ONNX (have to run onnxsim first on the ONNX file')
    parser.add_argument('-validate', type=bool, default=False, help='Validate that the outputs are similar enough to original outputs after optimization')
    args = parser.parse_args()
    n = args.n
    method = args.method
    device = args.device
    input_size = [n] + args.input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    model_file = args.model
    if args.simplified:
        model_file = model_file + "_simplified"

    # supply data size here? like with an if statement depending on model?
    print("Running inference on {0} random inputs with model {1} and method {2} on device {3}".format(n, model_file, method, device))
    ### take the outputs of the original pytorch model here to compare the outputs.
    ### NOTE assumes the original was made with pytorch.
    if method == "pytorch":
        from time_pytorch import run_pytorch_inference
        model, _ = utils.get_pytorch_model(args.model)
        outputs, inference_time = run_pytorch_inference(model, random_inputs, device)
    elif method == "tensorflow2":
        from time_tensorflow import run_tensorflow_inference
        from model_definitions.yolo_tf.models import (YoloV3, YoloV3Tiny)
        # TODO: make this change from code. also, would it in general be better to load model from code instead of file? small warnings when saving whole model with pt and tf
        model = utils.get_tensorflow_model(args.model)
        #Tensorflow uses (batch_size, H, W, C), pytorch (batch_size, C, H, W). for images
        if args.model in ["resnet50", "yolo", "squeezenet", "mobilenet", "ssd"]:
            random_inputs = np.swapaxes(random_inputs, 1, 3)
            random_inputs = np.swapaxes(random_inputs, 1, 2).astype(np.float32)
        outputs, inference_time = run_tensorflow_inference(model, random_inputs, device)
    elif method == "tensorrt":
        from time_tensorrt import run_tensorrt_inference
        # TODO?: change so tensorrt_inference doesnt need onnx just infer it from .trt path it should be same named?
        model_path = os.path.join(model_root_path, model_file, model_file + ".trt")
        outputs, inference_time = run_tensorrt_inference(model_path, random_inputs, os.path.join(model_root_path, model_file, model_file + ".onnx"))
    elif method == "openvino":
        from time_openvino import run_openvino_inference
        model_path = os.path.join(model_root_path, model_file, model_file + ".xml")
        outputs, inference_time = run_openvino_inference(model_path, random_inputs)
    elif method == "ngraph":
        from time_ngraph import run_ngraph_inference
        model_path = os.path.join(model_root_path, model_file, model_file + ".onnx")
        outputs, inference_time = run_ngraph_inference(model_path, random_inputs, device)
    elif method == "tensorflow":
        if device == "cpu":
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        from time_tensorflow import run_tensorflow1_inference
        model = utils.get_tensorflow1_graph_from_onnx(model_file)
        outputs, inference_time = run_tensorflow1_inference(model, random_inputs, "gpu")

    #pdb.set_trace()
    ## TODO: tensorrt outputs to single point in memory and all, fix this later!
    if args.validate:
        original_model, _ = utils.get_pytorch_model(args.model)
        original_outputs, _ = run_pytorch_inference(model, random_inputs)
        original_outputs = np.vstack([o.flatten().cpu().numpy() for o in original_outputs])
        outputs = np.vstack(outputs)
        np.testing.assert_allclose(original_outputs, outputs, rtol=1e-03, atol=1e-05)

    if args.save:
        utils.save_results(model_root_path, method, args.model, str(inference_time), args.n, device)
