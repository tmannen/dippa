"""
Test models using original pytorch and then some framework? Maybe tensorrt first?
"""

import numpy as np
import tensorrt as trt
import sys, os
import trt_common as common
import argparse
import utils
#import utils
from timeit import default_timer as timer

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

def get_engine(onnx_file_path, engine_file_path, input_size, rebuild=True):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    print("Explicit batch: ", common.EXPLICIT_BATCH)
    max_batch_size = 1
    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30 # 4096MB?
            builder.max_batch_size = max_batch_size
            #builder.int8_mode = True
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found.'.format(onnx_file_path))
                sys.exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print ('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print (parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [max_batch_size] + input_size
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)
            print("Completed creating Engine")
            #with open(engine_file_path, "wb") as f:
            #    f.write(engine.serialize())
            return engine
    """
    if os.path.exists(engine_file_path) and not rebuild:
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
    """
    return build_engine()

def run_tensorrt_inference(onnx_file_path, random_inputs, device):
    """
    Run inference on n random inputs? maybe define inputs later?
    """
    # Define inputs too? since some models have different shapes of inputs
    n = len(random_inputs)
    with get_engine(onnx_file_path, "", list(random_inputs.shape[1:])) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Do inference
        print('Running inference on {} random images'.format(n))
        trt_outputs = []
        times = []
        start = timer()
        # warmup
        for i in range(n//10):
            inputs[0].host = random_inputs[np.random.randint(0, n-1)]
            # [0] ok since all our models have just one output? (or many outputs in terms of scalars, but not in terms of layers)
            common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for i in range(n):
            inputs[0].host = random_inputs[i]
            # [0] ok since all our models have just one output? (or many outputs in terms of scalars, but not in terms of layers)
            prev = timer()
            out = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            trt_outputs.append(np.array(out[0]))
            times.append(timer() - prev)

        inference_time = (timer() - start)*1000 / n
        print(f'tensorrt inference time (msec): {inference_time:.5f}')
    # Before doing post-processing, we need to reshape the outputs as the common.do_inference will give us flat arrays.
    # print(trt_outputs[0].shape)
    return trt_outputs, times, inference_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-onnx_model_path', type=str, default=None, help='ONNX Model path.') # Example: "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.onnx"
    parser.add_argument('-trt_model_path', default="", type=str, help='TRT Model path.') # Example: "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.trt"
    parser.add_argument('-cpu', default=False, type=bool, help='Using CPU or not.')
    parser.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    parser.add_argument('-input_size', type=int, nargs='+', help='Input size (for ex. 3 224 224)', default=[3, 224, 224])
    # TODO: add -log boolean to args which defaults to False so the outputs are not logged if not wanted
    args = parser.parse_args()
    n = args.n
    input_size = [n] + args.input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    dir_path = "/".join(args.onnx_model_path.split("/")[:-1])
    name = args.onnx_model_path.split("/")[-1].split(".")[0]
    # supply data size here? like with an if statement depending on model?
    trt_outputs, inference_time = run_tensorrt_inference(args.trt_model_path, random_inputs, args.onnx_model_path)
    utils.save_results(dir_path, "tensorrt", name, str(inference_time), args.n)
    # main()
