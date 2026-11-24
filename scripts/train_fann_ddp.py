"""Launch with torchrun: complete-case multi-node DDP for faNN models 1 through 6."""
from datetime import timedelta
from tfai.benchmarks.fann_logging import print

import os
from pathlib import Path
from tfai.benchmarks.fann_comm_diagnostics import setup_debug,snapshot,configure_interop
# Set library debug filenames before importing Torch in each worker.
setup_debug()
import torch
import torch.distributed as dist
from scripts.train_fann import build_parser
from tfai.benchmarks.fann_startup_cache import load_startup
from tfai.benchmarks.fann_ddp import train_ddp,SUPPORTED


def main():
    parser=build_parser()
    parser.add_argument('--sampler',choices=['random','geometry_affinity'],default=os.environ.get('FANN_DDP_SAMPLER','random'))
    parser.add_argument('--backend',default=os.environ.get('FANN_DDP_BACKEND','nccl'))
    parser.add_argument('--timeout-minutes',type=int,default=120)
    parser.add_argument('--checkpoint-every',type=int,default=10)
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args()
    if len(args.models)!=1 or args.models[0] not in SUPPORTED:parser.error('Choose exactly one supported neural model (indices 1-6)')
    if args.epochs<1 or args.patience<1 or args.checkpoint_every<1 or args.timeout_minutes<1 or args.threads<1:parser.error('Counts must be positive')
    if args.max_updates is not None and args.max_updates<1:parser.error('max-updates must be positive')
    if args.learning_rate<=0 or not 0<args.min_lr<=args.learning_rate or args.grad_clip<=0:parser.error('Invalid learning rate/clipping')
    if args.cache_samples<1:parser.error('cache-samples must be positive')
    if args.graph_owner_nodes<1 or args.graph_edge_chunk<1 or args.graph_cache_mb<0:parser.error('Invalid graph chunk/cache setting')
    if os.environ.get('TURBINE_FANN_PIPELINE_GPUS','1')!='1':parser.error('DDP needs one full model per GPU; set TURBINE_FANN_PIPELINE_GPUS=1')
    local_rank=int(os.environ.get('LOCAL_RANK','0'));torch.set_num_threads(args.threads)
    configure_interop()
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        if local_rank>=torch.cuda.device_count():raise RuntimeError('LOCAL_RANK exceeds visible GPUs; check Slurm allocation and FANN_DDP_GPUS_PER_NODE')
        torch.cuda.set_device(local_rank);device=torch.device('cuda',local_rank)
    else:device=torch.device('cpu')
    communication_root=setup_debug()
    dist.init_process_group(args.backend,timeout=timedelta(minutes=args.timeout_minutes))
    rank=dist.get_rank()
    try:
        snapshot(communication_root,device,'initialized')
        packet=[None]
        if rank==0:
            try:
                manifest=load_startup(args.splits,args.allow_legacy_inputs,workers=int(os.environ.get('FANN_STARTUP_WORKERS','1')),
                                      progress=lambda message:print(message,flush=True))
                if args.domain!='legacy' and manifest['domain']!=args.domain:raise ValueError('Requested domain differs from prepared data')
                packet[0]={'manifest':manifest}
            except Exception as error:packet[0]={'error':str(error)}
        dist.broadcast_object_list(packet,src=0)
        if 'error' in packet[0]:raise ValueError(packet[0]['error'])
        excluded={'splits','models','output','resume','device','max_updates','allow_legacy_inputs','domain',
                  'backend','timeout_minutes','checkpoint_every','stop_file'}
        config={key:value for key,value in vars(args).items() if key not in excluded}
        name=args.models[0]
        result=train_ddp(name,packet[0]['manifest'],config,args.output/name,device,args.resume,args.max_updates,
                         args.checkpoint_every,args.stop_file)
        snapshot(communication_root,device,'completed')
        import resource,socket
        capacity={'rank':rank,'host':socket.gethostname(),'device':str(device),
                  'gpu_peak_GiB':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else None,
                  'process_peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
        capacities=[None]*dist.get_world_size();dist.all_gather_object(capacities,capacity)
        if rank==0:
            print('DDP capacity per rank: '+str(capacities),flush=True)
            print(f'{name}: DDP {result["state"]}; optimizer_steps={result.get("optimizer_steps",result.get("global_step"))}',flush=True)
            if result['state']=='interrupted':print('Safe DDP stop: last.pt saved; exit 0 so torchrun does not report a worker failure',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
