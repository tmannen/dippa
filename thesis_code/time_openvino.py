#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Teemu NOTE: need to run mo_onnx.py first. see notes.md'
 NOTE: inference originally from /opt openvino samples python classification sample?
"""

from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from timeit import default_timer as timer
import argparse
from openvino.inference_engine import IENetwork, IECore

def run_openvino_inference(model_xml, inputs, device="CPU"):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    log.info("Creating Inference Engine")
    ie = IECore()
    n = len(inputs)
    #if 'CPU' in device:
    #    ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)
    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 1
    #n, c, h, w = net.inputs[input_blob].shape
    #images = np.random.randn(n, c, h, w)
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=device)
    start = timer()
    ov_outputs = []
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    log.info("Starting inference in synchronous mode")
    for i in range(n):
        # [0] ok since all our models have just one output? (or many outputs in terms of scalars, but not in terms of layers)
        out = exec_net.infer(inputs={input_blob: inputs[i]})
        ov_outputs.append(out[out_blob])

    inference_time = (timer() - start)*1000 / n
    print(f'inference time (msec): {inference_time:.5f}')
    log.info("Batch size is {}".format(n))
    return ov_outputs, inference_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model_xml', type=str, default=None, help='XML Model path.') # requires model.bin (weights) in the same folder
    parser.add_argument('-device', default="CPU", type=str, help='Using CPU or not.')
    parser.add_argument('-n', default=1000, type=int, help='How many inputs to run the model on.')
    parser.add_argument('-input_size', type=int, nargs='+', help='Input size (for ex. 3 224 224)', default=[3, 224, 224])
    # TODO: add -log boolean to args which defaults to False so the outputs are not logged if not wanted
    args = parser.parse_args()
    n = args.n
    input_size = [n] + args.input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    dir_path = "/".join(args.model_xml.split("/")[:-1])
    name = args.model_xml.split("/")[-1].split(".")[0]
    # supply data size here? like with an if statement depending on model?
    ov_outputs, inference_time = run_openvino_inference(args.model_xml, args.device, random_inputs)
    #utils.save_results(dir_path, "openvino", name, str(inference_time), args.n)
    # main()