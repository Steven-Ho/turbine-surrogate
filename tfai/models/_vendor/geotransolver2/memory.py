"""Bound transient node-by-slice tensors while retaining global slice sums.

Only the deterministic, non-TE, non-plus path is supported. Parameters and
state-dict names are unchanged. Reduction order can differ in floating point.
"""
import torch
from torch.utils.checkpoint import checkpoint


def recompute(function,*args):
    if torch.is_grad_enabled():return checkpoint(function,*args,use_reentrant=False)
    return function(*args)


def chunked_apply(module,x,size):
    if not size or x.shape[1]<=size:return module(x)
    return torch.cat([recompute(module,part) for part in x.split(size,dim=1)],dim=1)


def context_slices(module,x,size):
    numerator=normalizer=None
    def aggregate(part):
        projected,features=module.project_input_onto_slices(part)
        weights=torch.softmax(module.in_project_slice(projected)/module.temperature.clamp(.5,5).to(part.dtype),dim=-1)
        value=weights.transpose(2,3)@features
        if torch.is_autocast_enabled(part.device.type):return value.float(),weights.float().sum(2)
        return value,weights.sum(2)
    for part in x.split(size,dim=1):
        value,total=recompute(aggregate,part)
        numerator=value if numerator is None else numerator+value
        normalizer=total if normalizer is None else normalizer+total
    return numerator/(normalizer[...,None]+1e-2)


def gale_attention(module,inputs,context,size):
    outputs=[]
    for x in inputs:
        numerator=normalizer=None
        def aggregate(part):
            projected,features=module.project_input_onto_slices(part)
            weights=torch.softmax(module.in_project_slice(projected)/module.temperature.clamp(.5,5).to(part.dtype),dim=-1)
            value=weights.permute(0,2,3,1)@features.permute(0,2,1,3)
            if torch.is_autocast_enabled(part.device.type):return value.float(),weights.float().sum(1)
            return value,weights.sum(1)
        for part in x.split(size,dim=1):
            value,total=recompute(aggregate,part)
            numerator=value if numerator is None else numerator+value
            normalizer=total if normalizer is None else normalizer+total
        slices=numerator/(normalizer[...,None]+1e-2)
        attended=module._compute_slice_attention_sdpa(slices)
        if context is not None:
            cross=module.compute_slice_attention_cross([slices],context)[0]
            mixing=module.state_mixing.sigmoid()
            attended=mixing*attended+(1-mixing)*cross
        def deslice(part,states):
            projected=module.in_project_x(part).reshape(part.shape[0],part.shape[1],module.heads,module.dim_head)
            weights=torch.softmax(module.in_project_slice(projected)/module.temperature.clamp(.5,5).to(part.dtype),dim=-1)
            return module._project_attention_outputs(states,weights)
        parts=[recompute(deslice,part,attended) for part in x.split(size,dim=1)]
        outputs.append(torch.cat(parts,dim=1))
    return outputs


def chunked_residual(module,x,size):
    if not size:return module(x)+x
    return torch.cat([recompute(lambda part:module(part)+part,part) for part in x.split(size,dim=1)],dim=1)
