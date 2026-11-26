"""Resume a validated small/8-node or base/16-node domain run to a total of 400 epochs."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def check(root,world=64,domain=4):
    folder=root/'GeoTransolver'
    for name in ['last.pt','best.pt','status.json','history.json']:
        if not (folder/name).is_file():raise ValueError(f'{root}: missing {name}')
    state=json.loads((folder/'status.json').read_text())
    if state.get('model')!='GeoTransolver' or state.get('world_size')!=world or state.get('domain_size')!=domain:
        raise ValueError(f'{root}: expected GeoTransolver with {world} ranks/domain_size={domain}')
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
    parser.add_argument('--preset',choices=['small8','base16'],default='small8')
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--dry-run',action='store_true')
    args,slurm=parser.parse_known_args()
    base=args.preset=='base16'
    nodes,domain,scale=(16,8,'base') if base else (8,4,'small')
    world=nodes*8
    pattern='fann_geo_base_16nodes_dom8_*' if base else 'fann_geo_domain4_8nodes_*'
    label='base16_long' if base else 'dom4_long'
    configured=os.environ.get('FANN_BASE_16_RUN_ROOT' if base else 'FANN_DOMAIN_RUN_ROOT')
    if args.run_root is None and configured:args.run_root=Path(configured)
    if args.run_root:
        root,epoch=check(Path(args.run_root),world,domain)
    else:
        candidates=[]
        for path in sorted(Path('runs').glob(pattern)):
            try:candidates.append(check(path,world,domain))
            except (ValueError,OSError,KeyError) as error:print(f'Skipping {error}',flush=True)
        if len(candidates)!=1:parser.error('Need exactly one interrupted domain run; specify --run-root runs/<original directory>')
        root,epoch=candidates[0]
    env=dict(os.environ,FANN_STABILITY_NODES=str(nodes),FANN_STABILITY_DOMAIN_SIZE=str(domain),FANN_STABILITY_SCALE=scale,
             FANN_STABILITY_LONG_RUN='1',FANN_STABILITY_EPOCHS='5',FANN_STABILITY_RESUME='1',
             FANN_STABILITY_LEARNING_RATE='0.0001',FANN_STABILITY_BETA1='0.9',FANN_STABILITY_BETA2='0.999',
             FANN_STABILITY_RUN_ROOT=str(root))
    env.pop('CUDA_VISIBLE_DEVICES',None)
    command=['bash','scripts/submit_fann_geo_stability.sh',
             f'--job-name=fann_geo_{label}',
             '--time='+os.environ.get('FANN_DOMAIN_TIME_LIMIT','1-00:00:00'),
             f'--output=logs/fann_geo_{label}_%A_%a.out',
             f'--error=logs/fann_geo_{label}_%A_%a.err',*slurm]
    print(f'Resume {root}: completed={epoch}, target=400 total epochs, remaining<={400-epoch}; {nodes} nodes/{world} GPUs, scale={scale}, 16 data groups x {domain} domain ranks',flush=True)
    print('Short-run budget disabled; scheduler and early stopping preserved; final test/export enabled upon training completion.',flush=True)
    if not args.dry_run:subprocess.run(command,env=env,check=True)


if __name__=='__main__':main()
