import torch
import math
import torch.nn.functional as F
from spikingjelly.activation_based.base import MemoryModule
from torch import Tensor, device, dtype
from .VN2 import VN2
from .PLIF import PLIF
from xs_snn.utils.override import Override
class VNext2(MemoryModule):
    '''
    VN
    '''
    duplicate=6
    id=0
    def __init__(self,small_id=0,duplicate=None,base_vth=0.6,vn_vth_offset=0.6,**kwargs):
        super().__init__()
        
        Override(VNext2,self,'duplicate',duplicate)

        self.base_vth=base_vth
        self.vn_vth_offset=vn_vth_offset
        
        # serial number
        self.id=VNext2.id+small_id
        VNext2.id+=1
        self.strength_group=VN2(dp=self.duplicate,base_vth=0.5,v_offset=vn_vth_offset)
        self.temperal_node=PLIF(base_vth=base_vth)

        self.reset()
        self.step_mode='s'  # multistep is in VN and PLIF, not here
        self.state_hooks=[]

    def reset(self):
        self.U_0=None
    
    def single_step_forward(self,input):

        o=self.strength_group(input) + self.temperal_node(input)

        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,input,o)
        return o

    def extra_repr(self):
        s = (f'id={self.id}')
        return s
