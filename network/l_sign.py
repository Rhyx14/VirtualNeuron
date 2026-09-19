from enum import Enum
class CLASSIFICATION(Enum):
    ImageNet=0
    CIFAR=1
    tinyImageNet=2
    SEG_BACKBONE=3

class FCN_MODE(Enum):
    FCN32s=0
    FCN16s=0
    FCN8s=0