from typing import Any, Callable
from x_secretary.utils.opencv_loader import OpenCV_Loader
from torchvision.transforms import v2
import random,logging,torchvision,torch
MEAN=(0.485, 0.456, 0.406)
STD=(0.229, 0.224, 0.225)

def get_imagenet1k_val_gpu(path,resize):
    '''
    get the imagenet-1k dataset with default val settings. 
    
    Data augmentation implemented on GPU

    @return: dataset, augmentation torch module 
    '''
    return torchvision.datasets.ImageFolder(path,
        transform=v2.Compose([
                v2.ToImage(),
                resize,
        ]),
    ),v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD)
    ])

def get_imagenet1k_train_gpu(path,resize,randomaugment_magnitude=9,erasing_possibility=0.5):
    '''
    get the imagenet-1k dataset with default training settings. 
    
    Data augmentation implemented on GPU

    @return: dataset, augmentation torch module 
    '''
    return torchvision.datasets.ImageFolder(path,
        transform=v2.Compose([
                v2.ToImage(),
        ]),
    ),v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        # v2.CutMix(alpha=1.0),
        # v2.MixUp(alpha=0.2),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(magnitude=randomaugment_magnitude),
        resize,
        v2.Normalize(MEAN,STD),
        v2.RandomErasing(p=erasing_possibility),
    ])