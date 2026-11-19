"""Run one formal model selected by the Slurm array index."""
import argparse,subprocess,sys
from pathlib import Path
from tfai.benchmarks.fann import MODELS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=Path,required=True);p.add_argument('--index',type=int,choices=range(7),required=True)
    p.add_argument('--epochs',type=int,default=400);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--domain',choices=['full','structured_patches'],default='full');p.add_argument('--resume',action='store_true')
    p.add_argument('--grid',type=int,default=32);p.add_argument('--rank',type=int,default=32)
    a=p.parse_args()
    command=[sys.executable,'scripts/train_fann.py','--device','cuda','--models',MODELS[a.index],
             '--splits',str(a.splits),'--epochs',str(a.epochs),'--output',str(a.output),'--domain',a.domain,
             '--grid',str(a.grid),'--rank',str(a.rank)]
    if a.resume:command.append('--resume')
    subprocess.run(command,check=True)

if __name__=='__main__':main()
