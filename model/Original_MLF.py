'''
Adapted from
https://github.com/langfengQ/MLF-DSResNet/blob/main/parallel_nets/spike_layer_for_cifar10.py
to spikingjelly version
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based.base import MemoryModule

# Vth = 0.6
# Vth2 = 1.6
# Vth3 = 2.6

TAU = 0.25
from .surrogate_functions import G_arctan
spikefunc=G_arctan.apply

class Original_MLF(MemoryModule):
    """ MLF unit.
    """
    id=0
    def __init__(self,duplicate=3,small_id=0,**kwds):
        super().__init__()
        self.step_mode='s'
        self.duplicate=duplicate
        self.vths=[0.6 + i for i in range(self.duplicate)]
        self.state_hooks=[]

        self.id=Original_MLF.id+small_id
        Original_MLF.id+=1

    def forward(self, x):
        frame_count,bs=x.shape[0],x.shape[1] # assert x.shape [t b c h w] or [t b c]
        u= [torch.zeros(x.shape[1:], device=x.device) for i in range(self.duplicate)]
        # u  = torch.zeros(x.shape[1:], device=x.device)
        # u2 = torch.zeros(x.shape[1:], device=x.device)
        # u3 = torch.zeros(x.shape[1:], device=x.device) # comment this line if you want MLF (K=2)
        o = torch.zeros(x.shape, device=x.device)
        for _t in range(frame_count):
            for i in range(self.duplicate):
                u[i] = TAU * u[i] * (1- spikefunc(u[i],self.vths[i]).detach()) + x[_t]
                _o=spikefunc(u[i],self.vths[i])
                o[_t,...] += _o

            # u = TAU * u * (1 - spikefunc(u,self.vths[0]).detach()) + x[_t, ...]
            # u2 = TAU * u2 * (1 - spikefunc(u2,self.vths[1]).detach()) + x[_t, ...]
            # u3 = TAU * u3 * (1 - spikefunc(u3,self.vths[2]).detach()) + x[_t, ...] # comment this line if you want MLF (K=2)
            # o[_t,...] = spikefunc(u,self.vths[0]) + spikefunc(u2,self.vths[1]) + spikefunc(u3,self.vths[2]) # Equivalent to union of all spikes
        if(len(self.state_hooks) != 0):
            for hooks in self.state_hooks:
                hooks(self.id,x,o)
        return o

    def extra_repr(self):
        return super().extra_repr() + f'duplicate (K)= {self.duplicate}, vth_base={self.vths[0]}'