"""
Test models using original pytorch and then some framework? Maybe tensorrt first?
"""

import numpy as np
import tensorrt as trt
import sys, os
import common
import argparse
from timeit import default_timer as timer

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    print("Explicit batch: ", common.EXPLICIT_BATCH)
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 28 # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 3, 224, 224]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
            return engine
    
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

def run_tensorrt_inference(onnx_file_path, engine_file_path, n=1000):
    """
    Run inference on n random inputs? maybe define inputs later?
    """
    # Define inputs too? since some models have different shapes of inputs
    outputs = []
    name = engine_file_path.split("/")[-1]
    with get_engine(onnx_file_path, engine_file_path) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on random images')
        start = timer()
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for _ in range(n):
            inputs[0].host = np.random.randn(1, 3, 224, 224).astype(np.float32)
            # [0] ok since all our models have just one output? (or many outputs in terms of scalar, but not in terms of layers)
            trt_outputs.append(common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)[0])

        print(f'{name} inference time (msec): {(timer() - start)*1000 / n:.5f}')
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    # print(trt_outputs[0].shape)
    return outputs

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-onnx_model_path', type=str, help='ONNX Model path.') # Example: "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.onnx"
    args.add_argument('-trt_model_path', type=str, help='TRT Model path.') # Example: "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.trt"
    args.add_argument('-cpu', default=False, type=bool, help='Using CPU or not.')
    args.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    # supply data size here? like with an if statement depending on model?
    parser = args.parse_args()
    run_tensorrt_inference(args.onnx_model_path, trt_model_path, args.n)
    # main()