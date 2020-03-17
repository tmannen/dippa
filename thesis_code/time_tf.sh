python time_models.py -method tensorflow -model resnet50 -device gpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model mobilenet -device gpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model squeezenet -device gpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model ssd -device gpu -save True -input_size 3 300 300
python time_models.py -method tensorflow -model fully_connected -device gpu -save True -input_size 784
python time_models.py -method tensorflow -model vgg16 -device gpu -save True -input_size 3 224 224

python time_models.py -method tensorflow -model resnet50 -device cpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model mobilenet -device cpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model squeezenet -device cpu -save True -input_size 3 224 224
python time_models.py -method tensorflow -model ssd -device cpu -save True -input_size 3 300 300
python time_models.py -method tensorflow -model fully_connected -device cpu -save True -input_size 784
python time_models.py -method tensorflow -model vgg16 -device cpu -save True -input_size 3 224 224