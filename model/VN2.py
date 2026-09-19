import torch
import math
import torch.nn.functional as F
from spikingjelly.activation_based.base import MemoryModule
from torch import Tensor, device, dtype
from xs_snn.utils.override import Override
class neuron_update(torch.autograd.Function):
    @staticmethod
    def forward(ctx,psp,vth,v_offset,dp):
        ctx.save_for_backward(psp)
        ctx.vth=vth
        ctx.v_offset=v_offset
        ctx.dp=dp
        return _firing(psp,vth,v_offset,dp)
    
    @staticmethod
    def backward(ctx,d_o):
        psp,=ctx.saved_tensors
        return _psp_backward(psp,d_o,ctx.vth,ctx.v_offset,ctx.dp),None,None,None

@torch.jit.script
def _firing(psp,vth:float,v_offset:float,dp:float):

    o=torch.div(psp-v_offset,vth,rounding_mode='floor')
    o=torch.clamp(o,0,dp) 

    return o

@torch.jit.script
def _psp_backward(psp, d_o,vth:float,v_offset:float,dp:float):

    alpha=2
    mult= 0.5* (torch.tanh(alpha*(psp-v_offset - vth/2))-torch.tanh(alpha*(psp-v_offset-dp*vth - vth/2))) / vth # g=0.5*tanh(a*x)+0.5
    # mult= torch.clamp_min(psp,0) - torch.clamp_min(psp-vth*256,0)
    # mult = 0.5 * (torch.exp(0.2 * psp)-torch.exp(0.2 * (psp-dp*vth))) / vth
    
    # mult= 1/3.14 * (torch.atan(3.14*psp)-torch.atan(3.14*(psp-dp*vth)))
    # mult= torch.gt(psp,0).type_as(psp)-torch.gt(psp-dp*vth,0).type_as(psp)

    grad= d_o * mult

    return grad

class VN2(MemoryModule):
    '''
    VN
    '''
    dp=6.
    base_vth=0.5
    v_offset=0.
    id=0
    update=neuron_update.apply

    def __init__(self,small_id=0,dp=None,base_vth=None,v_offset=None,**kwargs):
        super().__init__()
        self.step_mode='m'
        # serial number
        self.id=VN2.id+small_id
        VN2.id+=1
        
        Override(VN2,self,'base_vth',base_vth)
        Override(VN2,self,'dp',dp)
        Override(VN2,self,'v_offset',v_offset)
        self.reset()

        self.state_hooks=[]

    def reset(self):
        self.U_0=None
    
    def single_step_forward(self,input):
        if self.dp==0:
            return torch.zeros_like(input,device=input.device,dtype=input.dtype)
        
        o=VN2.update(input,self.base_vth,self.v_offset,self.dp)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,input,o)
        return o

    def extra_repr(self):
        s = (f'id={self.id},duplicate={self.dp},base_threshold={self.base_vth},v_offset={self.v_offset}')
        return s
