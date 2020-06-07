python time_models.py -method pytorch -model fully_connected -device cpu -save True -input_size 784
python time_models.py -method pytorch -model resnet50 -device cpu -save True -input_size 3 224 224
python time_models.py -method pytorch -model mobilenet -device cpu -save True -input_size 3 224 224
python time_models.py -method pytorch -model vgg16 -device cpu -save True -input_size 3 224 224
python time_models.py -method pytorch -model squeezenet -device cpu -save True -input_size 3 224 224
