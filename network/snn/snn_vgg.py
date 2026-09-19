from functools import partial
from typing import Any, cast, Dict, List, Optional, Union

import einops
import torch
import torch.nn as nn
from xs_snn.components import Repeat, Aggregated_Spiking_Layer,Identical_Wrapper
from spikingjelly.clock_driven.layer import SeqToANNContainer,MultiStepContainer

from ..l_sign import CLASSIFICATION
class VGG(nn.Module):
    def __init__(
        self, 
        frame_count,
        neuron_model,
        mode:CLASSIFICATION,
        features: nn.Module, 
        num_classes: int = 1000,
        name='VGG'
    ) -> None:
        super().__init__()

        self.name=name
        self.frame_count=frame_count

        self.stem=nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,bias=False),
            nn.BatchNorm2d(64),
            Repeat(pattern='b c h w -> t b c h w',t=self.frame_count),
            Aggregated_Spiking_Layer(None,None,neuron_model(inplane=64,frame_count=self.frame_count))
        )
        
        self.features = features
        match mode:
            case CLASSIFICATION.CIFAR:
                self.classifier = nn.Sequential(
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(512 * 2 * 2, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count)      
                    ),
                    # nn.Dropout(p=0.2),
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(4096, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count) 
                    ),
                )
            case CLASSIFICATION.ImageNet:
                self.classifier = nn.Sequential(
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(512 * 7 * 7, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count)      
                    ),
                    # nn.Dropout(p=0.2),
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(4096, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count) 
                    ),
                )
            case CLASSIFICATION.tinyImageNet: 
                self.classifier = nn.Sequential(
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(512 * 4 * 4, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count)      
                    ),
                    # nn.Dropout(p=0.2),
                    Aggregated_Spiking_Layer(
                        SeqToANNContainer(nn.Linear(4096, 4096)),
                        None,
                        neuron_model(frame_count=self.frame_count) 
                    ),
                )
            case _ : raise NotImplementedError

        self.final=nn.Linear(4096,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.stem(x)

        x = self.features(x)
        x= einops.rearrange(x,'t b c h w -> t b (c h w)')
        x = self.classifier(x)

        x= self.final(x.mean(0))
        return x

def make_layers(cfg: List[Union[str, int]], frame_count, neuron_model,batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 64
    for v in cfg:
        if v == "M":
            layers.append(MultiStepContainer(nn.AvgPool2d(kernel_size=2, stride=2)))
        elif v=="A":
            layers.append(MultiStepContainer(nn.AdaptiveAvgPool2d((7,7))))        
        else:
            v = cast(int, v)

            layers.append(
                Aggregated_Spiking_Layer(
                    SeqToANNContainer(nn.Conv2d(in_channels, v, kernel_size=3, padding=1,bias=False)),
                    batch_norm(v,frame_count=frame_count),
                    neuron_model(inplane=v,frame_count=frame_count)
                )
            )

            in_channels = v
    return nn.Sequential(*layers)

from ..vgg_settings import cfgs

def VGGs11(frame_count,neuron_model,num_classes,norm_layer,mode):
    return VGG(
        frame_count,
        neuron_model,
        mode,
        make_layers(cfgs[mode]['VGG11'], frame_count,neuron_model,batch_norm=norm_layer),
        num_classes=num_classes,
        name='VGGs11')

def VGGs13(frame_count,neuron_model,num_classes,norm_layer,mode):
    return VGG(
        frame_count,
        neuron_model,
        mode,
        make_layers(cfgs[mode]['VGG13'], frame_count,neuron_model,batch_norm=norm_layer),
        num_classes=num_classes,
        name='VGGs13')

def VGGs16(frame_count,neuron_model,num_classes,norm_layer,mode):
    return VGG(
        frame_count,
        neuron_model,
        mode,
        make_layers(cfgs[mode]['VGG16'], frame_count,neuron_model,batch_norm=norm_layer),
        num_classes=num_classes,
        name='VGGs16')

def VGGs19(frame_count,neuron_model,num_classes,norm_layer,mode):
    return VGG(
        frame_count,
        neuron_model,
        mode,
        make_layers(cfgs[mode]['VGG19'], frame_count,neuron_model,batch_norm=norm_layer),
        num_classes=num_classes,
        name='VGGs19')
