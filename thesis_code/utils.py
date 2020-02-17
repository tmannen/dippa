import pandas as pd
import numpy as np
import os
import csv
#import time_tensorrt
import time_pytorch
import torch
#import seaborn as sns
#import matplotlib as plt

def save_results(path, engine, model, time, n, device):
    # CSV with fields (engine, model, time, n)? appends to csv?
    save_path = os.path.join(path, "results.csv")
    fields = [engine, model, time, n, device]
    with open(save_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

def graph_results(results_path):
    # make a bar graph from results.csv? results has fields engine,model,time,n
    datas = pd.read_csv(results_path, header=None)
    datas.columns = ["method", "model", "time", "n", "device"] # Note: not needed if we add headers to csv
    datas['method'] = datas['method'] + " (" + datas['device'] + ")"
    sns.set(style="whitegrid")
    g = sns.catplot(x="model", y="time", hue="method", data=datas, kind="bar")
    g.set_ylabels("Average time per single inference(milliseconds)")
    plt.show()


def compare_accuracy_pt_trt(pt_model_path, trt_model_path, input_size, n=1000, onnx_model_path=None):
    # Needs the two models and what they're using (pytorch, tensorrt)? cool way would be to wrap in a predict() function and just call that
    # but maybe overkill. different function for each backend? pytorch vs tensorrt, pt vs openvino, etc.
    input_size = [n] + input_size
    random_inputs = np.random.randn(*input_size).astype(np.float32)
    pytorch_outputs = time_pytorch.run_pytorch_inference(pt_model_path, random_inputs)
    pytorch_outputs = torch.stack(pytorch_outputs).squeeze().cpu().numpy()
    # TRT outputs are the same if n > 1? probably because some memory copy shenanigans in tensorrt. see notes.md 24.01.2020
    trt_outputs = time_tensorrt.run_tensorrt_inference(trt_model_path, random_inputs)
    trt_outputs = np.vstack(trt_outputs).squeeze()
    np.testing.assert_allclose(pytorch_outputs, trt_outputs, rtol=1e-03, atol=1e-05)

def compare_accuracy(original_outputs, tool_outputs):
    np.testing.assert_allclose(original_outputs, tool_outputs, rtol=1e-03, atol=1e-05)

#compare_accuracy_pt_trt("/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.pt", 
#                        "/l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.trt", 
#                        [3, 224, 224], 2)