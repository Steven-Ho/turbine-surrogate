"""Submit capacity checks and six formal CR15 model jobs. Standard library only."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

RESOURCES={
 'POD':dict(gpus=1,cpus=8,memory='64G',scale='base',amp='off',hours=12),
 'UNet':dict(gpus=2,cpus=16,memory='64G',scale='small',amp='off',hours=12),
 'GCN':dict(gpus=4,cpus=32,memory='128G',scale='small',amp='off',hours=24),
 'Graph_UNet':dict(gpus=4,cpus=32,memory='128G',scale='small',amp='off',hours=24),
 'Graph_Transformer':dict(gpus=4,cpus=32,memory='128G',scale='small',amp='off',hours=24),
 'Transolver_3':dict(gpus=8,cpus=32,memory='128G',scale='base',amp='bf16',hours=12),
}


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--models',nargs='+',choices=list(RESOURCES),default=list(RESOURCES))
 p.add_argument('--prepared',type=Path,default=Path(os.environ.get('CR15_PREPARED','runs/cr15_prepared_v1')))
 p.add_argument('--run-root',type=Path)
 p.add_argument('--gpus',type=int,choices=[1,2,4,8],help='Override GPUs per model (single node, DDP)')
 p.add_argument('--epochs',type=int,default=400);p.add_argument('--epoch-budget',type=int,default=0)
 p.add_argument('--time-limit',help='Override Slurm wall time, e.g. 24:00:00')
 p.add_argument('--partition');p.add_argument('--account')
 p.add_argument('--resume',action='store_true',help='Resume last.pt in explicit --run-root; no capacity jobs')
 p.add_argument('--skip-capacity',action='store_true');p.add_argument('--dry-run',action='store_true')
 args=p.parse_args()
 if args.epochs<1 or args.epoch_budget<0:p.error('Invalid epoch budget')
 if args.resume and args.run_root is None:p.error('--resume requires --run-root')
 if len(args.models)!=len(set(args.models)):p.error('Duplicate models')
 root=(args.run_root or Path('runs')/('cr15_six_models_'+datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
 prepared=args.prepared.expanduser().resolve()
 if not args.dry_run:
  summary=json.loads((prepared/'summary.json').read_text())
  if summary.get('state')!='complete' or not summary.get('cache_ready'):raise ValueError('Prepared CR15 cache is not ready')
  Path('logs').mkdir(exist_ok=True);root.mkdir(parents=True,exist_ok=True)
 script=str(Path('scripts/slurm_cr15_train.sh').resolve());submissions=[]
 def submit(model,resources,capacity=False,dependency=None):
  env=os.environ.copy()
  env.pop('CUDA_VISIBLE_DEVICES',None)
  # Clear inherited experimental settings; CR15 normalizer/data and LR are shared.
  for key in list(env):
   if key.startswith('CR15_'):env.pop(key)
  gpus=1 if capacity else (args.gpus or resources['gpus'])
  run=root/('capacity_'+model) if capacity else root
  env.update(CR15_MODEL=model,CR15_PREPARED=str(prepared),CR15_RUN_ROOT=str(run),CR15_GPUS_PER_NODE=str(gpus),
   CR15_SCALE=resources['scale'],CR15_AMP=resources['amp'],CR15_BATCH_CASES='1' if capacity else '16',
   CR15_CAPACITY_CHECK='1' if capacity else '0',CR15_EPOCHS=str(args.epochs),CR15_EPOCH_BUDGET='0' if capacity else str(args.epoch_budget),
   CR15_RESUME='1' if args.resume and not capacity else '0',CR15_LARGEST_FIRST='1' if capacity else '0',
   CR15_LEARNING_RATE='0.0001',CR15_LR_PATIENCE='20',CR15_LR_THRESHOLD='0.001',CR15_LR_COOLDOWN='5',
   CR15_GRID='64',CR15_POD_RANK='32',CR15_GEOMETRY_BINS='256',
   CR15_MODEL_CACHE=str(Path('runs/cr15_model_cache_v1').resolve()),
   CR15_GRAPH_OWNER_NODES='32768',CR15_GRAPH_EDGE_CHUNK='131072')
  if capacity:env['CR15_MAX_UPDATES']='1'
  label='capacity' if capacity else 'train';prefix=f'cr15_{model}_{label}'
  command=['sbatch','--parsable','--nodes=1','--ntasks-per-node=1',f'--gres=gpu:{gpus}',
   f'--cpus-per-task={8 if capacity else resources["cpus"]}',f'--mem={resources["memory"]}',
   '--time='+('02:00:00' if capacity else (args.time_limit or f'{resources["hours"]}:00:00')),
   '--export=ALL','--job-name='+prefix,'--output=logs/'+prefix+'_%j.out','--error=logs/'+prefix+'_%j.err']
  if args.partition:command+=['--partition='+args.partition]
  if args.account:command+=['--account='+args.account]
  if dependency:command+=['--dependency=afterok:'+dependency]
  command.append(script)
  row=dict(model=model,phase=label,gpus=gpus,run_root=str(run),command=command,
           settings={k:v for k,v in env.items() if k.startswith('CR15_')})
  if args.dry_run:row['job_id']='DRY_'+model+'_'+label
  else:
   response=subprocess.run(command,env=env,text=True,capture_output=True,check=True)
   row['job_id']=response.stdout.strip().split(';')[0]
   if not row['job_id'].isdigit():raise ValueError('Unexpected sbatch job id: '+response.stdout)
  submissions.append(row)
  if not args.dry_run:(root/'submissions.json').write_text(json.dumps(submissions,indent=2)+'\n')
  print(json.dumps(row),flush=True);return row['job_id']
 for model in args.models:
  if args.resume and not (root/model/'last.pt').is_file() and not args.dry_run:raise FileNotFoundError(root/model/'last.pt')
  dependency=None
  if not args.skip_capacity and not args.resume:dependency=submit(model,RESOURCES[model],True)
  submit(model,RESOURCES[model],dependency=dependency)

if __name__=='__main__':main()
