## 03.10.2019

Kysymyksiä:

- tensorflow vs pytorch? especially tensorflow 2.0? just came out. pytorch more stable kinda?
- basics: train on beefy computer, use ONNX to translate to mobile devices, use best (eg. if original trained with pytorch, tensorflow might be better with mobile devices). Compare different approaches? 
- TVN and NNVM, alternatives to ONNX?
- what data is used in general? CARLA data? how to test several concurrent workloads, VMs or hwat?
- Maybe interesting: some common architecture in tensorflow and pytorch. test how fast it runs on 'native' tensorflow vs pytorch -> onnx -> tensorflow? how to make sure they're identical first, compare number of calculations? then if/when they're different, find out why and research this? what is onnx doing suboptimally?
- or: we want to compare compiling straight from onnx to some hardware? vs onnx -> tf/torch -> hardware?
- Interesting projects: Microsoft MMdnn? ONNX runtime (vs tensorflow native for example)
- utilisaatio, latenssit, gpu/cpu usage
- TODO: kokeile joku perus nn eka tensorflow, sitten pytorch -> onnx -> tensorflow. kumpi nopeampi?
- konvertoijat: https://github.com/ysh329/deep-learning-model-convertor ?

asenna softat:

- anaconda, tensorflow, pytorch, onnx (runtime?)

## 07.10.2019

### Installs:

- anaconda: anaconda.com, using .sh file
- pytorch: conda install pytorch torchvision cudatoolkit=10.0
- tensorflow: pip install tensorflow-gpu
- onnx: conda install -c conda-forge onnx
- onnx runtime: pip install onnxruntime (could also use pip for onnx..)

## 09.10.2019

- Intel ngraph? onnx tyylinen? 'compiler'?
- mittaa latency ja throughput?

## 10.10.2019

- get latex base from Anton or Vesa? ones found with google had some errors.
- Use VS code with plugins for latex?
- OPset version in torch onnx export? constant folding?

TODO: ?

## 11.10.2019

- what's the experiment setup? what ways to measure performance, source code, apis? nvidia, OS (utilization, latency, etc.)
- training different models on our dataset (CARLA?), compare results?
- nvidia nsight? other nvidia tools to measure performance?
- caffe2, pytorch, tf?
- jonkun kuvan ymmärtäminen tärkein: esim object recognition, segmentation? yolo network etc? kattele läpi ja ota paperit noteihin että mitä kaikkea löytyy.

### Notes

how to test neural network inference speeds reliably? run 1000 images and take time? how to make sure other doesnt lazyload or something? maybe run first one image to make sure model is 'loaded' and then take the real test? maybe should also test how fast it 'compiles' or something?

tried tutorial with resnet50 (more at thesis\_code/onnx\_runtime\_test.py): https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

would be really interesting to look into onnx translations and see if/where it's slower (did i already write this?)

netron visualizer?

TVM.ai? used here: https://github.com/dwofk/fast-depth, also has power consumption?

some kind of onnx compiler? https://skymizer.com/publications/Skymizer-AICAS2019.pdf

benchmarking? https://pdfs.semanticscholar.org/c462/ad5a425a4d0fcee178758a6782e7ba7d005b.pdf

^ found by searching "onnx" on google scholar

for research project: maybe something like a use case and then different ways of achieving it? for example for jetson, how to get our model on it and which is fastest? which is easiest to do? but nano is just a computer, everything works..? related: tensorRT? https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#network_api_pytorch_mnist

bigass article on object detection (only 2 months old): https://arxiv.org/pdf/1809.03193.pdf, newest reference from 2018 though?

hmm: https://www.reddit.com/r/MachineLearning/comments/8pcqgj/p_3d_object_detection_for_autonomous_driving/
hmm: https://www.reddit.com/r/MachineLearning/comments/a7iv7z/d_what_is_the_sota_for_3d_object_detection/
