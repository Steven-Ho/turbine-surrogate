"""Resume the validated 64-GPU domain run to a total of 400 epochs."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def check(root):
    folder=root/'GeoTransolver'
    for name in ['last.pt','best.pt','status.json','history.json']:
        if not (folder/name).is_file():raise ValueError(f'{root}: missing {name}')
    state=json.loads((folder/'status.json').read_text())
    if state.get('model')!='GeoTransolver' or state.get('world_size')!=64 or state.get('domain_size')!=4:
        raise ValueError(f'{root}: expected GeoTransolver with 64 ranks/domain_size=4')
    if state.get('state')!='interrupted':
        raise ValueError(f'{root}: expected safely interrupted run, got {state.get("state")}')
    epoch=state.get('completed_epochs')
    if not isinstance(epoch,int) or not 0<epoch<400:
        raise ValueError(f'{root}: expected 1..399 completed epochs')
    history=json.loads((folder/'history.json').read_text())
    if len(history)!=epoch or history[-1].get('epoch')!=epoch:
        raise ValueError(f'{root}: history does not match completed epochs')
    return root,epoch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path,default=os.environ.get('FANN_DOMAIN_RUN_ROOT'))
    parser.add_argument('--dry-run',action='store_true')
    args,slurm=parser.parse_known_args()
    if args.run_root:
        root,epoch=check(Path(args.run_root))
    else:
        candidates=[]
        for path in sorted(Path('runs').glob('fann_geo_domain4_8nodes_*')):
            try:candidates.append(check(path))
            except (ValueError,OSError,KeyError) as error:print(f'Skipping {error}',flush=True)
        if len(candidates)!=1:parser.error('Need exactly one interrupted domain run; specify --run-root runs/<original directory>')
        root,epoch=candidates[0]
    env=dict(os.environ,FANN_STABILITY_NODES='8',FANN_STABILITY_DOMAIN_SIZE='4',
             FANN_STABILITY_LONG_RUN='1',FANN_STABILITY_EPOCHS='5',FANN_STABILITY_RESUME='1',
             FANN_STABILITY_LEARNING_RATE='0.0001',FANN_STABILITY_BETA1='0.9',FANN_STABILITY_BETA2='0.999',
             FANN_STABILITY_RUN_ROOT=str(root))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    command=['bash','scripts/submit_fann_geo_stability.sh',
             '--job-name=fann_geo_dom4_long',
             '--time='+os.environ.get('FANN_DOMAIN_TIME_LIMIT','1-00:00:00'),
             '--output=logs/fann_geo_dom4_long_%A_%a.out',
             '--error=logs/fann_geo_dom4_long_%A_%a.err',*slurm]
    print(f'Resume {root}: completed={epoch}, target=400 total epochs, remaining<={400-epoch}; 8 nodes/64 GPUs, 16 data groups x 4 domain ranks',flush=True)
    print('Short-run budget disabled; scheduler and early stopping preserved; final test/export enabled upon training completion.',flush=True)
    if not args.dry_run:subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
