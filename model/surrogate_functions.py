import torch
class G_arctan(torch.autograd.Function):
    """ 
    近似梯度 arctan
    """
    @staticmethod
    def forward(ctx, spsp,vth):
        ctx.save_for_backward(spsp)
        ctx.vth=vth
        output = torch.gt(spsp, vth) 
        return output.type_as(spsp)

    @staticmethod
    def backward(ctx, dy):
        spsp, = ctx.saved_tensors 
        return _back(spsp, dy,ctx.vth),None
        # frac=(3.1416 /2 * alpha)**2
        # hu = alpha / (2 * (1 + input**2 * frac))
        # return dy * hu

@torch.jit.script
def _back(spsp,dy,vth:float):
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (spsp-vth)**2 * frac))
    return dy*hu

class G_rect(torch.autograd.Function):
    """ 
    近似梯度 rectangle
    """
    @staticmethod
    def forward(ctx, input, vth):
        ctx.save_for_backward(input)
        ctx.vth=vth
        output = torch.gt(input, vth)
        return output.type_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        a=1.0
        grad_input = grad_output.clone()
        hu = (abs(input - ctx.vth) < (a/2)) / a
        return grad_input * hu,None

class LIF2_dynamic(torch.autograd.Function):
    """ 
    LIF update, with arctan surrogate gradient
    """
    @staticmethod
    def forward(ctx, s_psp,vth,tau):
        ctx.save_for_backward(s_psp)
        ctx.vth=vth
        ctx.tau=tau
        output,U= _lif_forward(s_psp,vth,tau)
        return output.type_as(s_psp), U

    @staticmethod
    def backward(ctx, dy,du):
        s_psp, = ctx.saved_tensors 
        return _lif_back(s_psp, dy,du,ctx.vth,ctx.tau),None,None
        # frac=(3.1416 /2 * alpha)**2
        # hu = alpha / (2 * (1 + input**2 * frac))
        # return dy * hu

# @torch.jit.script
def _lif_forward(s_psp:torch.Tensor,vth:float,tau:float):
    output = torch.gt(s_psp, vth).type_as(s_psp)
    U=  tau * (1-output) * s_psp
    return output,U

@torch.jit.script
def _lif_back(s_psp,dy,du,vth:float,tau:float):
    alpha=2
    frac=(3.1415926 /2 * alpha)**2
    hu = alpha / (2 * (1 + (s_psp-vth)**2 * frac))
    do=dy*hu # gradient from surrogate, space

    _o=torch.gt(s_psp,vth).type_as(du)
    dt= tau * du * (1-_o) # gradient from membrane potential, time
    return do+dt
