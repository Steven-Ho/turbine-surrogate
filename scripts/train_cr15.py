"""Train GeoTransolver on prepared A-grade CR15 five-field caches."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import torch
import torch.distributed as dist
from tfai.benchmarks.cr15 import load_prepared
from tfai.benchmarks.cr15_training import train


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('runs/cr15_prepared_v1'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--backend')
    p.add_argument('--scale',choices=['small','base','large'],default='base')
    p.add_argument('--amp',choices=['off','bf16'],default='bf16')
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--epoch-budget',type=int,default=0)
    p.add_argument('--max-updates',type=int)
    p.add_argument('--batch-cases',type=int,default=16)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--largest-first',action='store_true')
    p.add_argument('--offload',action='store_true')
    p.add_argument('--chunk',type=int,default=32768)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--beta1',type=float,default=.9);p.add_argument('--beta2',type=float,default=.999)
    p.add_argument('--lr-patience',type=int,default=20);p.add_argument('--lr-factor',type=float,default=.5)
    p.add_argument('--lr-threshold',type=float,default=.001);p.add_argument('--lr-cooldown',type=int,default=5)
    p.add_argument('--min-lr',type=float,default=1e-6);p.add_argument('--patience',type=int,default=60)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--checkpoint-every',type=int,default=10);p.add_argument('--log-every',type=int,default=1)
    p.add_argument('--stop-file',type=Path)
    args=p.parse_args()
    if min(args.epochs,args.batch_cases,args.chunk,args.threads,args.checkpoint_every,args.log_every,args.patience)<1:p.error('Counts must be positive')
    if args.epoch_budget<0 or args.max_updates is not None and args.max_updates<1:p.error('Invalid stopping budget')
    if not 0<=args.beta1<1 or not 0<=args.beta2<1 or not 0<args.min_lr<=args.learning_rate or args.grad_clip<=0:p.error('Invalid optimizer setting')
    if args.seed<0:p.error('seed must be nonnegative')
    if args.lr_patience<0 or args.lr_cooldown<0 or not 0<args.lr_factor<1 or args.lr_threshold<0:p.error('Invalid LR scheduler settings')
    torch.set_num_threads(args.threads)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent CPU fallback')
        local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:device=torch.device('cpu')
    world=int(os.environ.get('WORLD_SIZE','1'))
    if world>1:dist.init_process_group(args.backend or ('nccl' if device.type=='cuda' else 'gloo'),timeout=timedelta(minutes=120))
    try:
        if world>1:
            packet=[None]
            if dist.get_rank()==0:
                try:packet[0]=dict(prepared=load_prepared(args.prepared))
                except Exception as error:packet[0]=dict(error=str(error))
            dist.broadcast_object_list(packet,src=0)
            if 'error' in packet[0]:raise ValueError(packet[0]['error'])
            prepared=packet[0]['prepared']
        else:prepared=load_prepared(args.prepared)
        excluded=['prepared','output','device','backend','resume','epoch_budget','max_updates','stop_file']
        config={k:v for k,v in vars(args).items() if k not in excluded}
        train(prepared,config,args.output/'GeoTransolver',device,args.resume,args.epoch_budget,args.max_updates,args.stop_file)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
