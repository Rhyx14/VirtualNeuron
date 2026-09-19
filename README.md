# Training multi-bit Spiking Neural Network with Virtual Neurons
An implementation of the Virtual Neurons of the paper 
```BibTex
@article{XU2025129825,
title = {Training multi-bit Spiking Neural Network with Virtual Neurons},
journal = {Neurocomputing},
volume = {634},
pages = {129825},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.129825},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225004977},
author = {Haoran Xu and Zonghua Gu and Ruimin Sun and De Ma},
}
```

The defination of various spiking neuron are in neuron/

```
VNext2 -> VN
PLIF -> LIF with learnable \tau
OriginalMLF -> MLF
```

# Training 

1. install packages in the requirements.txt 

2. install two additional .whl in the pkg/

3. Launch the training on the tiny-ImageNet dataset via
```
python main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1
```

or in DDP mode with 4 GPUs
```
torchrun --nproc_per_node=4 main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1 --gpu 0,1,2,3
```