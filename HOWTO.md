# Basic setup needed for every tool:

- Install python (version used in testing was 3.7). Testing install was done with anaconda.
- Run pip install requirements.txt. requirements.txt file in thesis_code folder (TODO)
- run export_models.py with the model you want to use. Example:

```
python export_models.py -model_name resnet50 -opset 9
```

- Time models after exporting with time_models.py. Note: some tools need more preparation. See tool specific instructions below. Example:

```
python time_models.py -method openvino -model resnet50 -device cpu
```

## ONNX

- ONNX files are exported with the export_models.py script
- *Optional:* Run onnx-simplifier. Example:

```
python -m onnxsim /l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.onnx /l/dippa_main/dippa/thesis_code/models/resnet50/resnet50_simplified.onnx
```

### TensorRT

- Install TensorRT with python following instructions in: https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar

### OpenVINO

- Install Openvino following instructions in: http://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_linux.html
- Note: running ```source /opt/intel/openvino/bin/setupvars.sh```  is required before using openVINO. You can add this to .bashrc if you always want to setup the correct variables.
- OpenVINO requires using their model optimizer script before using it. Example run:

```
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model /l/dippa_main/dippa/thesis_code/models/resnet50/resnet50.onnx --output_dir /l/dippa_main/dippa/thesis_code/models/resnet50
```

### nGraph

- nGraph is used with ONNX and is installed when installing requirements.txt