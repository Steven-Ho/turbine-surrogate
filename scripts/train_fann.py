"""Formal faNN training: explicit domain, geometry holdouts, resume and frozen test."""
import argparse
import json
import os
from pathlib import Path
import torch
from tfai.benchmarks.fann import MODELS
from tfai.benchmarks.fann_training import train_one, atomic_json, RunLockedError
from tfai.benchmarks.fann_startup_cache import load_startup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',choices=['full','structured_patches','legacy'],default='full')
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=400)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-legacy-inputs',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--lr-patience',type=int,default=8)
    p.add_argument('--lr-factor',type=float,default=.5)
    p.add_argument('--lr-threshold',type=float,default=.005)
    p.add_argument('--lr-cooldown',type=int,default=2)
    p.add_argument('--min-lr',type=float,default=1e-6)
    p.add_argument('--patience',type=int,default=60)
    p.add_argument('--grid',type=int,default=32)
    p.add_argument('--rank',type=int,default=32)
    p.add_argument('--scale',choices=['small','base','large'],default='small')
    p.add_argument('--pod-geometry-bins',type=int,default=256)
    p.add_argument('--streaming-pod',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--grad-clip',type=float,default=1.)
    p.add_argument('--near-wall',type=float,default=.001)
    p.add_argument('--cache-samples',type=int,default=1)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--graph-execution',choices=['dense','bounded'],default='bounded')
    p.add_argument('--graph-owner-nodes',type=int,default=32768)
    p.add_argument('--graph-edge-chunk',type=int,default=262144)
    p.add_argument('--graph-cache-mb',type=int,default=1024)
    p.add_argument('--max-updates',type=int,help='Stop and save after this many additional updates; recovery verification')
    a=p.parse_args()
    if a.epochs<1 or a.patience<1 or a.rank<1 or a.pod_geometry_bins<1 or a.grad_clip<=0 or a.learning_rate<=0 or not 0<a.min_lr<=a.learning_rate:
        p.error('Invalid budget, rank, clipping or learning-rate setting')
    if a.cache_samples<1 or a.threads<1 or (a.max_updates is not None and a.max_updates<1):p.error('Counts must be positive')
    if a.graph_owner_nodes<1 or a.graph_edge_chunk<1 or a.graph_cache_mb<0:p.error('Invalid graph chunk/cache size')
    torch.set_num_threads(a.threads)
    if a.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no silent CPU fallback')
    print(f'faNN startup: models={a.models} splits={a.splits} device={a.device}',flush=True)
    manifest=load_startup(a.splits,a.allow_legacy_inputs,workers=int(os.environ.get('FANN_STARTUP_WORKERS','1')),
                          progress=lambda message:print(message,flush=True))
    print('faNN split inspection complete; entering model initialization',flush=True)
    if a.domain!='legacy' and manifest['domain']!=a.domain:
        p.error('Requested domain differs from prepared data: '+manifest['domain'])
    config={key:value for key,value in vars(a).items() if key not in ['splits','models','output','resume','device','max_updates','allow_legacy_inputs','domain']}
    errors={}
    for model in a.models:
        folder=a.output/model
        try:
            result=train_one(model,manifest,config,folder,a.device,a.resume,a.max_updates)
            if result['state']=='interrupted':
                peak=([torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())]
                      if a.device.startswith('cuda') else None)
                print(f'{model}: stopped safely after {result["global_step"]} updates; last.pt saved; per_gpu_peak_allocated_GiB={peak}',flush=True)
                raise SystemExit(130)
        except RunLockedError as error:
            print(str(error),flush=True);errors[model]=str(error)
        except Exception as error:
            import traceback
            text=traceback.format_exc();print(text,flush=True);errors[model]=text
            atomic_json({'state':'failed','model':model,'error':str(error)},folder/'status.json')
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
