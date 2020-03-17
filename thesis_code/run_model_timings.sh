python time_models.py -method tensorrt -model $1 -device gpu -save True -input_size $2
python time_models.py -method pytorch -model $1 -device gpu -save True -input_size $2
python time_models.py -method pytorch -model $1 -device cpu -save True -input_size $2
echo "Running openvino model optimizer:"
python /opt/intel/openvino/deployment_tools/model_optimizer/mo_onnx.py --input_model /l/dippa_main/dippa/thesis_code/models/$1/$1.onnx --output_dir /l/dippa_main/dippa/thesis_code/models/$1
python time_models.py -method openvino -model $1 -device cpu -save True -input_size $2
python time_models.py -method ngraph -model $1 -device cpu -save True -input_size $2
echo "Remember to run python time_models.py -method tensorflow -model $1
-device gpu -save True -input_size \"$2\" in tf_gpu environment if you want tensorflow results"
