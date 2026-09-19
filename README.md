<a id="english"></a>

# Training multi-bit Spiking Neural Network with Virtual Neurons

[English](#english) | [中文](#中文)

An implementation of the Virtual Neurons from the paper

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

The definitions of various spiking neurons are in neuron/

```
VNext2 -> VN
PLIF -> LIF with a learnable \tau
OriginalMLF -> MLF
```

# Training 

1. Install the packages in requirements.txt.

2. Install the two additional .whl files in pkg/.

3. Launch training for the tiny-ImageNet dataset via
```
python main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1
```

or in DDP mode with 4 GPUs
```
torchrun --nproc_per_node=4 main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1 --gpu 0,1,2,3
```

---

<a id="中文"></a>

# Training multi-bit Spiking Neural Network with Virtual Neurons

[English](#english) | [中文](#中文)

论文中虚拟神经元（Virtual Neurons）的实现

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

各类脉冲神经元的定义位于 neuron/ 目录下

```
VNext2 -> VN
PLIF -> LIF with a learnable \tau
OriginalMLF -> MLF
```

# 训练

1. 安装 requirements.txt 中的依赖包。

2. 安装 pkg/ 目录下的两个额外 .whl 文件。

3. 通过以下命令启动 tiny-ImageNet 数据集的训练
```
python main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1
```

或使用 4 块 GPU 的 DDP 模式
```
torchrun --nproc_per_node=4 main_train_timg.py --neuron VN --arch SEW2s18 --FRAME_COUNT 1 --gpu 0,1,2,3
```