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