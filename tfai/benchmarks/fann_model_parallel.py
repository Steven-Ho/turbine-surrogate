"""Single-process, within-node layer model parallelism for full faNN cases.

This distributes parameters and saved activations, not the batch. It does not
promise throughput scaling: a single case visits stages sequentially.
"""
from tfai.benchmarks.fann_logging import print

import os
import torch

SUPPORTED={'GCN','Graph_UNet','Graph_Transformer','GeoTransolver'}


def move(value,device):
    if torch.is_tensor(value):return value.to(device)
    if isinstance(value,tuple):return tuple(move(v,device) for v in value)
    if isinstance(value,list):return [move(v,device) for v in value]
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    return value


def stage(module,device):
    module.to(device)
    def transfer(_module,args,kwargs):return move(args,device),move(kwargs,device)
    module.register_forward_pre_hook(transfer,with_kwargs=True)


def configure(model,device,count=None):
    count=int(os.environ.get('TURBINE_FANN_PIPELINE_GPUS','1')) if count is None else int(count)
    if count<1:raise ValueError('TURBINE_FANN_PIPELINE_GPUS must be positive')
    if count==1:return model
    if model.name not in SUPPORTED:raise ValueError('Layer model parallelism supports '+str(sorted(SUPPORTED)))
    if model.name=='Graph_UNet' and getattr(model,'graph_executor',None) is None:
        raise ValueError('Graph_UNet layer parallelism requires --graph-execution bounded')
    primary=torch.device(device)
    if primary.type!='cuda':raise ValueError('Layer model parallelism requires CUDA-compatible GPUs')
    first=primary.index if primary.index is not None else torch.cuda.current_device()
    if first+count>torch.cuda.device_count():raise ValueError('Insufficient visible GPUs for requested pipeline')
    devices=[torch.device('cuda',first+i) for i in range(count)]
    primary=devices[0];net=model.net
    if model.name=='GCN':
        stages=list(net.processor)
        for i,block in enumerate(stages):stage(block,devices[min(count-1,i*count//len(stages))])
        stage(net.node_decoder,devices[-1])
    elif model.name=='Graph_Transformer':
        for i,(conv,norm) in enumerate(zip(net.convs,net.norms)):
            d=devices[min(count-1,i*count//len(net.convs))];stage(conv,d);stage(norm,d)
        stage(net.output_proj,devices[-1])
    elif model.name=='GeoTransolver':
        core=net.model
        for i,block in enumerate(core.blocks):stage(block,devices[min(count-1,i*count//len(core.blocks))])
        for output in core.ln_mlp_out:stage(output,devices[-1])
    else:
        core=net.net;layers=list(core.down_convs)+list(core.up_convs)
        for i,layer in enumerate(layers):layer.to(devices[min(count-1,i*count//len(layers))])
        for i,pool in enumerate(core.pools):pool.to(next(core.down_convs[i+1].parameters()).device)
    # Training targets and external callers keep their existing primary device.
    net.register_forward_hook(lambda _module,_args,output:move(output,primary))
    model.pipeline_devices=devices
    print(f'{model.name}: layer model parallel devices={devices}; sequential stages, no DDP',flush=True)
    return model
