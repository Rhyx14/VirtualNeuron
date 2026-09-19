from typing import Dict,List
from .l_sign import CLASSIFICATION
cfgs_imagenet: Dict[str, List[ str |  int]] = {
    "VGG11": ["M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "A"],
    "VGG13": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "A"],
    "VGG16": [64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "A"],
    "VGG19": [64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "A"],
}

cfgs_cifar: Dict[str, List[ str |  int]] = {
    "VGG11": ["M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG13": [64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512],
    "VGG16": [64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512],
    "VGG19": [64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512],
}

cfgs={
    CLASSIFICATION.CIFAR:cfgs_cifar,
    CLASSIFICATION.ImageNet:cfgs_imagenet,
    CLASSIFICATION.tinyImageNet:cfgs_cifar
    }
