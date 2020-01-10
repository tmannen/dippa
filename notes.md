## 03.10.2019

Kysymyksiä:

- tensorflow vs pytorch? especially tensorflow 2.0? just came out. pytorch more stable kinda?
- basics: train on beefy computer, use ONNX to translate to mobile devices, use best (eg. if original trained with pytorch, tensorflow might be better with mobile devices). Compare different approaches? 
- TVM and NNVM, alternatives to ONNX?
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
- caffe2, pytorch, tf onnx runtime?
- jonkun kuvan ymmärtäminen tärkein: esim object recognition, segmentation? yolo network etc? kattele läpi ja ota paperit noteihin että mitä kaikkea löytyy.

## 14.10.2019

- https://blogs.nvidia.com/blog/2018/08/10/autonomous-vehicles-perception-layer/:
    - "These networks include DriveNet, which detects obstacles, and OpenRoadNet, which detects drivable space. To plan a path forward, LaneNet detects lane edges and PilotNet detects drivable paths."
- Maybe depth and 3d vision could be the more complicated ones? also the object counting thing.
- Maybe end-to-end will be simpler and better?
- Mask-RCNN for object detection?
- use pretrained layer and finetune on CARLA data - only last layer finetuning or whole network?
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
- "this study uses two(or just one?) different object detection/segmentation models trained on CARLA dataset and measures the performance and accuracy when converted to ONNX? and using different ONNX translators (and using the target software) and runtime(s)? Also compares to native implementation?"

## 15.10.2019

- Problem with using ONNX on x86: might not be fair since pytorch and tf are so optimized for that platform already, while onnx could provide speed ups/even enable use at all on mobile or embedded systems?
- mentioned already but more support for them: https://ai.facebook.com/blog/onnx-expansion-speeds-ai-development-/ . Hardware runtimes "such as NVIDIA’s TensorRT and Intel’s nGraph" (mostly for mobile? not sure)
- https://cloudblogs.microsoft.com/opensource/2019/08/26/announcing-onnx-runtime-0-5-edge-hardware-acceleration-support/ "ONNX Runtime 0.5 with support for edge hardware acceleration", sounds relevant?
- any good place to ask someone what problems they have with different platfors, maybe edge computing relating to ML?
- IoT edge computing azure, aws?
- *concentrate on the self driving car (Steering, gas?), train model(s) on CARLA data, test? Jyris paper probably relevant here. <- for research project*
- different pratforms: intel integrated GPUs? (for dippa?)

## 17.10.2019

- Taulukko/feature matriisi että mitä mitkäkin komponentit tekevät (tensorRT, openwino, onnx runtime jne.)
    - ota lista eri työkaluista ja katso mitä ne osaa/mihin tarkoitukseen ne on. esim. inference, iGPU, CPU, mitä formaattia osaa, voiko train / design vai ei. 
- esim tensorRT vain inferenssi. vs tensorflow, kumpi nopeampi?
- throughput, latency, resurssien kulutus (muisti jne.) mittaa eri komponentteja etta missä mikin parempi. eri alustoja myös? esim. integratee gpu, jetson, jne.

## 18.10.2019

- ONNX runtime python needs to be installed specifically for GPU? pip install onnxruntime-gpu. different backends, different installations? (use C++ instead?)

## 22.10.2019

- ONNX runtime cpu - slow? also not using MKL, how to use? How to use GPU? MKL apparently not supported for python, have to use C++/C#?
- a lot of different things to think about in latency: cache, loading from memory to GPU vs. just CPU, have to 'warmup' the runs to get accurate results? What if we use C++ and python, are their timers the same?
- TODO: kysy jaakolta mikä paras tapa runnia c++? :V C++ tarvii varmaan onnx runtime MKL jne.

## 24.10.2019

- Tried onnxruntime-gpu (new conda env, scuffed_latency_test.py), the difference between that and pytorch native eval is almost non existent.
- installed cudnn with the ubuntu debs, gets installed in /usr/include? is it used then? (well at least the error onnxruntime-gpu ran into disappeared after that)
    - installed ALSO with just libraries - will it break something? libcudnn didn't seem to be in the correct place so had to do it.

- installed tensorrt 6.0: https://developer.nvidia.com/nvidia-tensorrt-6x-download
    - install notes: used .deb file, followed the sudo dpkg -i etc. install seems to be in /usr/src? NOT ANYMORE, INSTALLED FROM TAR
- installing onnxruntime (wheel?) from source:
    - cmake needed to be updated: used official apt, worked (google it)
    - cloned github onnxruntime repo. install command: `./build.sh --cudnn_home /usr/include --cuda_home /usr/local/cuda --use_tensorrt --tensorrt_home /usr/src/tensorrt/ --build_wheel --update`
    - didnt work, updated locale, try --update fix here?: https://github.com/microsoft/onnxruntime/issues/1203:
        - `locale-gen en_US.UTF-8 update-locale LANG=en_US.UTF-8`
    - got error: `Error: 'libpython3.7m.so.1.0: cannot open shared object file`. fixed by running: `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/81/karkkat5/unix/anaconda3/envs/onnxruntime-trt/lib`
    - Apparently a debug build - problem?

## 28.10.2019

- Got onnxruntime-trt working. Seems to bring an improvement to the inference! roughly a 30% improvement compared to pytorch in eval mode (also against onnxruntime-gpu, which had almost identical times to pytorch)
- TODO: try building with ngraph, openvino, mkldnn/mklml?
- TODO: check that the results are correct pytorch vs tensorrt (or at least close enough)
- TODO?: onnxruntime build with all backends and choose in script? is it possible?

## 31.10.2019

- reinforcement learning CARLA simulation? steering?
- steering test problem: if only one camera the car doesn't know how to correct if it steers away fro mthe middle of the road?
- carla simulation data needs data where the car recovers from mistakes?

## 16.12.2019

- TODO: onnxruntime tensorrt uusiks joku cublas bullshitti -_- testaa cublas toimivuus? jotain 10.1 eri paikassa ehkä cudassa blaa

## 17.12.2019

- installed cudnn again: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html (7.6.5)
- tensorrt installation: https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar
- some cublas errors (not necessarily because tensorrt?): reinstalled cuda, ran purge cuda, clean, jne with sudo. LD library path properly, added to .bashrc
- TODO: try docker images?
- build instructions for onnx runtime: https://github.com/microsoft/onnxruntime/blob/master/BUILD.md
- TODO: onnxruntime reinstall: ./build.sh --cudnn_home /usr/include --cuda_home /usr/local/cuda --use_tensorrt --tensorrt_home /l/software/TensorRT-6.0.1.5 --build_wheel --update --build --enable_pybind | all of these needed?
- had to update LD_LIBRARY_PATH like so: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/u/81/karkkat5/unix/anaconda3/envs/tensorrt/lib. Needs to be in LD_LIBRARY_PATH for onnxruntime, otherwise it complais cant find some libpython3.7? NOTE: added the conda env tensorrt lib.

TODO: more than tensorrt, maybe build onnxruntime with all if possible? python support for all possible? more about interoperability?

## 19.12.2019

- Dissecting nvidia vola gpu architecture via microbenchmarking (paperi) Zhe jia, marco maggioni
- performance testing in shell mode?
- onnxruntime perftest important - TODO: pb data?. pb data created with create_random_data.py

## 30.12.2019

Dippa notes (image in gmail):
- Interoperability benchmarking?
- docker? forgot to ask
- many different models for testing, cnns mostly, nlp. export to onnx, successful? if not, this is experiment data itself. test this onnx translation with many backends/tools?
- WHY is some optimizer better? table of runtimes, models. some runtime might surprise, explain why it's bad/good. quantization, etc. is it still similar enough to the original model? statistical things, mean, std. does run time vary between runs?
- ONNXruntime performancetest - is it enough? what about tool specific optimizations that the onnxruntime tool might not support? maybe test them by themselves too, see how they fare?
- Profiling link: https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9339-profiling-deep-learning-networks.pdf
- TODO: pb data edelleen? ota pic gmailista. kokeile converter monilla malleilla?

## 03.01.2020

- onnx_perf_test: main logic in ort_test_session.cc? performance_runner.cc imports from there?
- Installing openvino:
    - https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux download
    - follow instructions: http://docs.openvinotoolkit.org/2019_R3.1/_docs_install_guides_installing_openvino_linux.html
    - (install prerequisites jne. also changed $HOME to export HOME=/l (do permanently?) due to permissions issues with sample.) ran the sample with sudo
    - especially "source /opt/intel/openvino/bin/setupvars.sh (maybe put in .bashrc?)" NOTE remember to do this before building? if using openvino

- Installing ngraph (no need for onnx runtime? but needed if I want to do it myself?) TODO?
- Big build command using most backends: ```./build.sh --config RelWithDebInfo --cudnn_home /usr/include --cuda_home /usr/local/cuda --use_tensorrt --tensorrt_home /l/software/TensorRT-6.0.1.5 --use_dnnl --use_ngraph --use_openvino GPU_FP32 --build_wheel --update --build --enable_pybind```

## 07.01.2020

- TODO: (test docker tensorrt, is docker enough when performance is key?), test speeds? run perf_test in docker and normally, see if any inefficiencies?
- TODO?: write script to create docker, run perf test? commands, upload them to local, etc.?
- try openvino tool by itself?
- TODO: what to write to thesis?

## 09.01.2020

- TODO: write more about the models chosen and why, eg. are they using batch norm and so on?
- TODO: actually try them out in tensorrt maybe, also test if onnx export works even. (check that model outputs are sufficiently close)
- maybe yolo in model_defs instead of models or something?

## 10.01.2020

- PYTORCH many errors when saving model: UserWarning: Couldn't retrieve source code for container of type AdaptiveAvgPool2d. It won't be checked for correctness upon loading. resnet50
- TENSORRT: kirjoita että vaikeampi saada käyntiin, tarttee kaikkee allocate buffers n shit.
- TENSORRT: started tensorrt_test.py that loans code from yolov3_onnx tensorrt repo. uses common.py in samples and PYCUDA. some problems, like bad error messages (saying just invalid arguments although the dtype of the image was wrong, needed to switch from fp64 to fp32). need to manually allocate memory and stuff? what about using int8 optimizations and things like that?
- TODO maanantai: jatka tensorrt kikkailua, kokeile accuracy sama, ehkä more optimizations. vertaa onnx runtime? 

# Misc Notes

how to test neural network inference speeds reliably? run 1000 images and take time? how to make sure other doesnt lazyload or something? maybe run first one image to make sure model is 'loaded' and then take the real test? maybe should also test how fast it 'compiles' or something?

tried tutorial with resnet50 (more at thesis\_code/onnx\_runtime\_test.py): https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html

would be really interesting to look into onnx translations and see if/where it's slower (did i already write this?)

netron visualizer?

TVM.ai? used here: https://github.com/dwofk/fast-depth, also has power consumption?

some kind of onnx compiler? https://https://pytorch.org/tutorials/intermediate/torchvision_tutorial.htmlskymizer.com/publications/Skymizer-AICAS2019.pdf
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
benchmarking? https://pdfs.semanticshttps://pytorch.org/tutorials/intermediate/torchvision_tutorial.htmlcholar.org/c462/ad5a425a4d0fcee178758a6782e7ba7d005b.pdf
https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
^ found by searching "onnx" on google scholar

for research project: maybe something like a use case and then different ways of achieving it? for example for jetson, how to get our model on it and which is fastest? which is easiest to do? but nano is just a computer, everything works..? related: tensorRT? https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#network_api_pytorch_mnist

bigass article on object detection (only 2 months old, though it was just edited then): https://arxiv.org/pdf/1809.03193.pdf, newest reference from 2018 though?

hmm: https://www.reddit.com/r/MachineLearning/comments/8pcqgj/p_3d_object_detection_for_autonomous_driving/
hmm: https://www.reddit.com/r/MachineLearning/comments/a7iv7z/d_what_is_the_sota_for_3d_object_detection/
hmm: https://www.reddit.com/r/MachineLearning/comments/bb53uo/foveabox_beyond_anchorbased_object_detector/
hmm: https://github.com/open-mmlab/mmdetection (toolbox for object detection, supports many sota implementations)

https://developer.nvidia.com/drive/drive-networks <- many different approaches to nn self driving cars

https://medium.com/syncedreview/deep-learning-in-real-time-inference-acceleration-and-continuous-training-17dac9438b0b <- some hardware stuff about inference optimization - also good point about models improving fast and the need to continuously train it again? Also, software optimization like network pruning.

https://software.intel.com/en-us/articles/optimization-practice-of-deep-learning-inference-deployment-on-intel-processors - openvino and intel inference optimization stuff.

https://medium.com/moonvision/onnx-the-long-and-collaborative-road-to-machine-learning-portability-a6416a96e870 - onnx frameworks, compilers, so on.

seems like a very relevant link: https://towardsdatascience.com/benchmarking-hardware-for-cnn-inference-in-2018-1d58268de12a
    - "1) The Intel Neural Stick is 4x faster than than the Google Vision Kit, which both use the same underlying Movidius 2450 board. Software implementations of the same layers matter."

very similar paper from jussi hanhirova? https://arxiv.org/pdf/1803.09492.pdf

use cases for edge computing:
    - liikennevalot ml siellä? huono lähettää kuvaa cloudiin jne.

cool stuff: https://microsoft.github.io/onnxruntime/auto_examples/plot_profiling.html

