"""Standard-library preflight: resume the successful 16-rank CR15 run to 400."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def check(root):
    root=Path(root).expanduser().resolve()
    if root.name=='GeoTransolver':root=root.parent
    folder=root/'GeoTransolver'
    for name in ['last.pt','best.pt','status.json','history.json','configuration.json']:
        if not (folder/name).is_file():raise ValueError(f'{root}: missing {name}')
    status=json.loads((folder/'status.json').read_text())
    configuration=json.loads((folder/'configuration.json').read_text());config=configuration['config']
    if status.get('dataset')!='CR15' or status.get('world_size')!=16 or status.get('global_batch_cases')!=16:
        raise ValueError(f'{root}: expected CR15 world_size=16 and global batch=16')
    if status.get('state')!='interrupted':raise ValueError(f'{root}: expected safely interrupted run, got {status.get("state")}')
    epoch=status['completed_epochs']
    if not isinstance(epoch,int) or not 1<=epoch<=400:raise ValueError(f'{root}: invalid completed epochs')
    expected=dict(scale='base',amp='bf16',batch_cases=16,chunk=32768,largest_first=False,offload=False,
                  beta1=.9,beta2=.999,seed=42,grad_clip=1.,checkpoint_every=10,log_every=1)
    for key,value in expected.items():
        if config.get(key)!=value:raise ValueError(f'{root}: unsupported saved configuration {key}={config.get(key)}')
    history=json.loads((folder/'history.json').read_text())
    if len(history)!=epoch or history[-1]['epoch']!=epoch:raise ValueError(f'{root}: history/epoch mismatch')
    return root,epoch,configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=os.environ.get('CR15_TWO_NODE_ROOT'))
    parser.add_argument('--prepared',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args,slurm=parser.parse_known_args()
    if args.run_root:
        root,epoch,configuration=check(args.run_root)
    else:
        matches=[]
        for path in sorted(Path('runs').glob('cr15_geo_base_2nodes_1epoch_*')):
            try:matches.append(check(path))
            except (ValueError,OSError,KeyError) as error:print(f'Skipping {error}',flush=True)
        if len(matches)!=1:parser.error('Specify --run-root: need exactly one safely interrupted two-node CR15 experiment')
        root,epoch,configuration=matches[0]
    config=configuration['config']
    env=dict(os.environ,CR15_RUN_ROOT=str(root),CR15_PREPARED=str(args.prepared or os.environ.get('CR15_PREPARED') or configuration['prepared_root']),
             CR15_SCALE='base',CR15_AMP='bf16',CR15_BATCH_CASES='16',CR15_GPUS_PER_NODE='8',
             CR15_RESUME='1',CR15_EPOCHS='400',CR15_EPOCH_BUDGET='0',CR15_LARGEST_FIRST='0',
             CR15_LEARNING_RATE=str(config['learning_rate']),CR15_LR_PATIENCE=str(config['lr_patience']),
             CR15_LR_THRESHOLD=str(config['lr_threshold']),CR15_LR_COOLDOWN=str(config['lr_cooldown']))
    # These settings are CLI defaults in the shared Slurm launcher. Refuse a
    # mismatch before allocating GPUs, rather than silently resetting a scheduler.
    for key,value in [('lr_factor',.5),('min_lr',1e-6),('patience',60)]:
        if config.get(key)!=value:parser.error(f'Unsupported saved {key}; use a matching direct CLI resume')
    env.pop('CR15_MAX_UPDATES',None);env.pop('CUDA_VISIBLE_DEVICES',None)
    command=['sbatch','--exclusive','--nodes=2','--ntasks-per-node=1','--gres=gpu:8',
             '--cpus-per-task=32','--mem=64G','--time='+os.environ.get('CR15_TIME_LIMIT','1-00:00:00'),
             '--export=ALL','--job-name=cr15_2n_long','--output=logs/cr15_2n_long_%j.out',
             '--error=logs/cr15_2n_long_%j.err',*slurm,'scripts/slurm_cr15_train.sh']
    print(f'CR15 resume: {root}; epoch={epoch}, total target=400, remaining<={400-epoch}; 2 nodes x 8 GPUs, batch=16',flush=True)
    print('Per-launch epoch/update limits disabled. Optimizer/scheduler/RNG restored; final validation/test/baseline evaluation enabled.',flush=True)
    if not args.dry_run:
        Path('logs').mkdir(exist_ok=True)
        subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
