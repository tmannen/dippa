python time_models.py -method tensorrt -model $1 -device gpu -save True -input_size $2
python time_models.py -method pytorch -model $1 -device gpu -save True -input_size $2
python time_models.py -method pytorch -model $1 -device cpu -save True -input_size $2
python time_models.py -method openvino -model $1 -device cpu -save True -input_size $2
python time_models.py -method ngraph -model $1 -device cpu -save True -input_size $2
python time_models.py -method tensorflow -model $1 -device gpu -save True -input_size $2
