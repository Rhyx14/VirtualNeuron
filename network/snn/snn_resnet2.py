import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from torchvision.models.resnet import conv1x1,conv3x3 

# from xs_snn.components import Aggregated,Aggregated_Spiking_Layer,Identical_Wrapper
from xs_snn.components import Aggregated_Spiking_Layer,Identical_Wrapper, Repeat
from spikingjelly.clock_driven.layer import SeqToANNContainer,MultiStepContainer
class MS_BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        neuron_model,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iter_id=0,
        layer_id=0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.iter_id=iter_id
        self.residue=True
        self.layer_id=layer_id

        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1_s=Aggregated_Spiking_Layer(
            None,
            None,
            neuron_model(inplane=planes),name='sn_first')
        
        self.conv1=Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(inplanes, planes, stride,groups=groups)),
            norm_layer(planes),
            neuron_model(inplane=planes))
        
        self.conv2= Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(planes, planes,groups=groups)),
            norm_layer(planes))

        if downsample is not None:
            # conv + bn
            assert isinstance(downsample,Aggregated_Spiking_Layer)
            downsample._neuron_model=None
            self.res_path=downsample
        else:
            self.res_path=norm_layer(planes)

        self.empty_wrapper=Identical_Wrapper(name='shortcut')
        self.output_wrapper=Identical_Wrapper(name='layer_output')
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.output_wrapper(self.conv1_s(x))
        out = self.conv1(out)
        out = self.conv2(out)

        if self.residue:
            identity=self.empty_wrapper(self.res_path(x))
            out += identity

        return out
    
    def extra_repr(self):
        s = (f'layer_id={self.layer_id},iter_id={self.iter_id},residue={self.residue}')
        return s

class SN_BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        neuron_model,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iter_id=0,
        layer_id=0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.iter_id=iter_id
        self.residue=True
        self.layer_id=layer_id

        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1=Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(inplanes, planes, stride,groups=groups)),
            norm_layer(planes),
            neuron_model(inplane=planes))
        
        self.conv2= Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(planes, planes,groups=groups)),
            norm_layer(planes),
            None)
        
        self.conv2_s=Aggregated_Spiking_Layer(
            None,
            None,
            neuron_model(inplane=planes),name='sn_last')

        if downsample is not None:
            # conv + bn
            assert isinstance(downsample,Aggregated_Spiking_Layer)
            downsample._neuron_model=None
        self.downsample=downsample
        self.empty_wrapper=Identical_Wrapper(name='shortcut')
        self.output_wrapper=Identical_Wrapper(name='layer_output')
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.residue:
            identity=self.empty_wrapper(identity)
            if self.downsample is not None:
                identity = self.downsample(identity)
            
            out += identity

        out=self.output_wrapper(self.conv2_s(out))

        return out
    
    def extra_repr(self):
        s = (f'layer_id={self.layer_id},iter_id={self.iter_id},residue={self.residue}')
        return s

class SEW_BasicBlock(nn.Module):
    expansion: int = 1
    def __init__(
        self,
        neuron_model,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        iter_id=0,
        layer_id=0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.iter_id=iter_id
        self.residue=True
        self.layer_id=layer_id

        if base_width != 64:
            raise ValueError('BasicBlock only supports base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1=Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(inplanes, planes, stride,groups=groups)),
            norm_layer(planes),
            neuron_model(inplane=planes),name='res_first')
        
        self.conv2=Aggregated_Spiking_Layer(
            SeqToANNContainer(conv3x3(planes, planes,groups=groups)),
            norm_layer(planes),
            neuron_model(inplane=planes),name='res_last')

        self.downsample = downsample # conv + bn
        self.output_wrapper=Identical_Wrapper(name='layer_output')
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.residue:
            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
        out = self.output_wrapper(out)
        return out

    def extra_repr(self):
        s = (f'layer_id={self.layer_id},iter_id={self.iter_id},residue={self.residue}')
        return s
from ..l_sign import CLASSIFICATION
class ResNet(nn.Module):

    def __init__(
        self,
        block,
        layers: List[int],
        neuron_model,
        frame_count,
        mode:CLASSIFICATION,
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        name='resnet'
    ) -> None:
        super(ResNet, self).__init__()

        self.name=name
        self.frame_count=frame_count
        self._norm_layer = norm_layer
        self._neuron_model=neuron_model
        self.layers_config=layers
        self.block_type=block

        self.inplanes = 64

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        match mode:
            case CLASSIFICATION.CIFAR | CLASSIFICATION.tinyImageNet:
                self.stem=nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,padding=1,bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    Repeat(pattern='b c h w -> t b c h w', t=self.frame_count),
                    Aggregated_Spiking_Layer(None,None,neuron_model(inplane=self.inplanes))
                )
            case CLASSIFICATION.ImageNet:
                self.stem=nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=4,padding=3,bias=False),
                    nn.BatchNorm2d(self.inplanes),
                    Repeat(pattern='b c h w -> t b c h w', t=self.frame_count),
                    Aggregated_Spiking_Layer(None,None,neuron_model(inplane=self.inplanes))
                )
            case CLASSIFICATION.SEG_BACKBONE:
                self.stem=nn.Sequential(
                    nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,padding=3,bias=False), # load ImageNet weight but use this conv
                    nn.BatchNorm2d(self.inplanes),
                    Repeat(pattern='b c h w -> t b c h w', t=self.frame_count),
                    Aggregated_Spiking_Layer(None,None,neuron_model(inplane=self.inplanes))
                )
            case _ : raise NotImplementedError


        self.layer1 = self._make_layer(block, 64, layers[0],layer_id=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],layer_id=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],layer_id=3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],layer_id=4)

        if self.block_type is MS_BasicBlock:
            self.last_sn=Aggregated_Spiking_Layer(None,None,neuron_model(inplane=512))
        else:
            self.last_sn=Identical_Wrapper(name='placeholder')

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=torch.nn.Linear(512*block.expansion,num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (norm_layer)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        for m in self.modules():
            if isinstance(m, SEW_BasicBlock):
                nn.init.constant_(m.conv2._norm.weight, 0) 
            if isinstance(m, (SN_BasicBlock,)):
                nn.init.constant_(m.conv2._norm.weight, 0) 
            
            # if isinstance(m, (SEW_Bottleneck)):
            #     nn.init.constant_(m.conv3._norm.weight, 0)
            # if isinstance(m, (SN_Bottleneck,)):
            #     nn.init.constant_(m.norm_3.weight, 0) 
  
    def _make_layer(self,block, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False,
                    layer_id=1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Aggregated_Spiking_Layer(
                    MultiStepContainer(
                        nn.AvgPool2d(kernel_size=stride,stride=stride),
                        nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,stride=1,bias=False)                    
                    ),
                    norm_layer(planes * block.expansion),
                    self._neuron_model(inplane=planes*block.expansion,small_id=0.1))

        layers = []
        layers.append(block(self._neuron_model,self.inplanes, planes,stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer,
                            iter_id=1,layer_id=layer_id,frame_count=self.frame_count))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self._neuron_model,self.inplanes, planes,groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer,
                                iter_id=i+1,
                                layer_id=layer_id,
                                frame_count=self.frame_count))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:

        x=self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x= self.last_sn(x)
        
        x=self.avgpool(x.mean(0))
        x=x.flatten(1)  # [b,c,1,1] -> [b,c]
        x=self.fc(x) 
        return x

def SEW2s18(**kwargs: Any) -> ResNet:
    return ResNet(SEW_BasicBlock, [2, 2, 2, 2], name='SEW2s18', **kwargs)

def SEW2s34(**kwargs: Any) -> ResNet:
    return ResNet(SEW_BasicBlock, [3, 4, 6, 3], name='SEW2s34', **kwargs)

def SR2s18(**kwargs: Any) -> ResNet:
    return ResNet(SN_BasicBlock, [2, 2, 2, 2], name='SR2s18',**kwargs)

def SR2s34(**kwargs: Any) -> ResNet:
    return ResNet(SN_BasicBlock, [3, 4, 6, 3],name='SR2s34',**kwargs)

def MS2s18(**kwargs: Any) -> ResNet:
    return ResNet(MS_BasicBlock, [3, 4, 6, 3], name='MS2s18', **kwargs)

def MS2s34(**kwargs: Any) -> ResNet:
    return ResNet(MS_BasicBlock, [3, 4, 6, 3], name='MS2s34', **kwargs)


