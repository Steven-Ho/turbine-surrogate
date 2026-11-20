"""Register faNN once on CPU; all following models reuse checks and train stats."""
import argparse
import os
from pathlib import Path
import torch
from tfai.benchmarks.fann_startup_cache import load_startup


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=int(os.environ.get('FANN_STARTUP_WORKERS','1')))
    parser.add_argument('--cache-dir',type=Path)
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--force',action='store_true',help='Recheck all contents, even with a valid registration')
    group.add_argument('--from-checkpoint',type=Path,help='Reuse previous checks/stats; explicitly asserts prepared data unchanged since this checkpoint')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    torch.set_num_threads(1)
    manifest=load_startup(args.splits,cache_dir=args.cache_dir,workers=args.workers,force=args.force,
                          from_checkpoint=args.from_checkpoint,progress=lambda message:print(message,flush=True))
    print(f'Registration ready: signature={manifest["signature"]} domain={manifest["domain"]}',flush=True)


if __name__=='__main__':main()
