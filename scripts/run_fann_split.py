"""Run one model using fixed geometry-disjoint train/validation splits."""
import argparse,json,subprocess,sys
from pathlib import Path
from tfai.benchmarks.fann import MODELS
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--splits',type=Path,required=True);p.add_argument('--index',type=int,choices=range(7),required=True)
p.add_argument('--steps',type=int,default=20000);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();s=json.loads(a.splits.read_text());model=MODELS[a.index]
subprocess.run([sys.executable,'scripts/run_fann.py','--device','cuda','--models',model,'--train',*s['train'],
 '--valid',*s['valid'],'--steps',str(a.steps),'--output',str(a.output/model)],check=True)
