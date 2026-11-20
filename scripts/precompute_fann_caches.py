"""Prebuild static faNN artifacts for all seven models and record timings."""
import argparse
import os
from pathlib import Path
import time
import torch
from tfai.benchmarks.fann import MODELS
from tfai.benchmarks.fann_startup_cache import load_startup
from tfai.benchmarks.fann_precompute import precompute


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,default=Path(os.environ.get('FANN_SPLITS','runs/fann_prepared_20261119_202344/splits.json')))
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--output',type=Path,default=Path('runs/fann_cache_precompute'))
    p.add_argument('--device',default='cuda')
    p.add_argument('--scale',choices=['small','base','large'],default='small')
    p.add_argument('--grid',type=int,default=int(os.environ.get('FANN_GRID','32')))
    p.add_argument('--rank',type=int,default=int(os.environ.get('FANN_POD_RANK','32')))
    p.add_argument('--owner-nodes',type=int,nargs='+',default=[int(os.environ.get('FANN_GRAPH_OWNER_NODES','32768'))])
    p.add_argument('--graph-edge-chunk',type=int,default=int(os.environ.get('FANN_GRAPH_EDGE_CHUNK','262144')))
    p.add_argument('--graph-cache-mb',type=int,default=0)
    p.add_argument('--workers',type=int,default=int(os.environ.get('FANN_STARTUP_WORKERS','1')))
    p.add_argument('--threads',type=int,default=int(os.environ.get('OMP_NUM_THREADS','4')))
    p.add_argument('--pod-checkpoint',type=Path,help='Reuse a matching old POD basis rather than fitting it again')
    args=p.parse_args()
    if min(args.grid,args.rank,args.graph_edge_chunk,args.workers,args.threads,*args.owner_nodes)<1 or args.graph_cache_mb<0:p.error('Invalid counts')
    if 'UNet' in args.models and (args.grid<32 or args.grid%16):p.error('UNet grid must be >=32 and divisible by16')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU fallback for GPU cache identity')
    torch.set_num_threads(args.threads)
    started=time.perf_counter()
    manifest=load_startup(args.splits,workers=args.workers,progress=lambda message:print(message,flush=True))
    startup_seconds=time.perf_counter()-started
    config={key:getattr(args,key) for key in ('scale','grid','rank','owner_nodes','graph_edge_chunk','graph_cache_mb')}
    report=precompute(manifest,list(dict.fromkeys(args.models)),config,args.output,args.device,args.pod_checkpoint,startup_seconds)
    print('Cache preparation complete: '+str(args.output/'report.json'),flush=True)
    for name,row in report['models'].items():
        print(f'{name}: wall_seconds={row["wall_seconds"]:.3f} built_tasks={row["built_tasks"]} cache_hits={row["cache_hits"]}',flush=True)
    print(f'Unique model artifact size: {report["unique_artifact_bytes"]/2**30:.3f} GiB',flush=True)


if __name__=='__main__':main()
