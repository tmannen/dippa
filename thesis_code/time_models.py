"""
Create a single file for timing models. Import different inferences 
from other places (like run_tensorrt_inference) and just control what tool to use with an if statement here
"""

import argparse
import numpy as np
import os
import utils
from time_tensorrt import run_tensorrt_inference
from time_pytorch import run_pytorch_inference
from time_ngraph import run_ngraph_inference
import pdb
## This is for the YOLO model:
import sys
sys.path.append('model_definitions/')

model_root_path = "/l/dippa_main/dippa/thesis_code/models/"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default=None, help='Model name, for example "resnet50".')
    parser.add_argument('-method', type=str, help='pytorch, tensorrt, ngraph or openvino')
    parser.add_argument('-device', type=str, default="gpu", help='cpu or gpu')
    parser.add_argument('-save', default=False, type=bool, help='Wheter to save the results to a CSV or not')
    parser.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    parser.add_argument('-input_size', type=int, nargs='+', help='Input size (for ex. 3 224 224)', default=[3, 224, 224])

    args = parser.parse_args()
    n = args.n
    method = args.method
    device = args.device
    input_size = [n] + args.input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    model = args.model
    # supply data size here? like with an if statement depending on model?

    ### take the outputs of the original pytorch model here to compare the outputs.
    ### NOTE assumes the original was made with pytorch.
    original_model_path = os.path.join(model_root_path, model, model + ".pt")
    original_outputs, _ = run_pytorch_inference(original_model_path, random_inputs)
    if method == "pytorch":
        model_path = os.path.join(model_root_path, model, model + ".pt")
        outputs, inference_time = run_pytorch_inference(model_path, random_inputs, device)
    elif method == "tensorrt":
        model_path = os.path.join(model_root_path, model, model + ".trt")
        outputs, inference_time = run_tensorrt_inference(model_path, random_inputs, os.path.join(model_root_path, model, model + "_simplified.onnx"))
    elif method == "openvino":
        from time_openvino import run_openvino_inference
        model_path = os.path.join(model_root_path, model, model + ".xml")
        outputs, inference_time = run_openvino_inference(model_path, random_inputs)
    elif method == "ngraph":
        model_path = os.path.join(model_root_path, model, model + ".onnx")
        outputs, inference_time = run_ngraph_inference(model_path, random_inputs, device)

    original_outputs = np.vstack([o.flatten().cpu().numpy() for o in original_outputs])
    outputs = np.vstack(outputs)
    #pdb.set_trace()
    ## TODO: tensorrt outputs to single point in memory and all, fix this later!
    np.testing.assert_allclose(original_outputs, outputs, rtol=1e-03, atol=1e-05)
    if args.save:
        utils.save_results(model_root_path, method, model, str(inference_time), args.n, device)
