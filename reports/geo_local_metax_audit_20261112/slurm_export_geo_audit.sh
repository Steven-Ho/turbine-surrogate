#!/bin/bash
#SBATCH --job-name=geo_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --get-user-env
#SBATCH --output=logs/geo_audit_%j.out
#SBATCH --error=logs/geo_audit_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export TFAI_PROJECT_ROOT="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export GEO_AUDIT_RUN="${GEO_AUDIT_RUN:-runs/metax_geo_kdtree_1455}"
srun --ntasks=1 --gres=gpu:1 python - <<'PY'
import csv,hashlib,json,os,sys,tarfile
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
sys.path.insert(0,str(Path.cwd()/'scripts'))
from compare_models import read_vtu,metrics
root=Path(os.environ['GEO_AUDIT_RUN']).resolve();run=root/'GeoTransolver'
out=root/('geo_audit_'+os.environ.get('SLURM_JOB_ID','manual'));out.mkdir(exist_ok=False)
checkpoint=torch.load(run/'train/best_model.pt',map_location='cpu',weights_only=False)
def plain(v):
    if isinstance(v,torch.Tensor):return v.tolist()
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [plain(x) for x in v]
    if v is None or isinstance(v,(str,int,float,bool)):return v
    return str(v)
def save(name,value):
    (out/name).write_text(json.dumps(plain(value),indent=2,ensure_ascii=False)+'\n')
save('checkpoint_config.json',checkpoint['config'])
save('checkpoint_norm_params.json',checkpoint['norm_params'])
norm=run/'data/processed/norm_params.pt'
if norm.exists():save('disk_norm_params.json',torch.load(norm,map_location='cpu',weights_only=False))
splits={}
for split in ('train','valid'):
    table=run/f'data/processed/effective_{split}.csv'
    if not table.exists():raise FileNotFoundError(f'Exact effective split unavailable: {table}')
    rows=list(csv.DictReader(table.open()))
    splits[split]=[{**r,'file_path':Path(r['file_path']).name} for r in rows]
test=list(csv.DictReader((run/'input/test.csv').open()))
splits['test']=[{**r,'file_path':Path(r['file_path']).name} for r in test]
save('effective_splits.json',splits)
evaluation=checkpoint['evaluate_result_test'];entries=evaluation['samples_evaluated']
assert len(entries)==100
baseline=json.loads((Path.cwd()/'configs/cluster_long/local_baseline.json').read_text())
threshold=baseline['mape_absolute_threshold_pa'];result=[];names=[]
for entry in entries:
    source=Path(entry['source_file']);prediction=Path(entry['evaluated_result_path'])
    xyz,true,topology=read_vtu(source,True)
    pxyz,pred,ptopology=read_vtu(prediction,True)
    np.testing.assert_array_equal(xyz,pxyz)
    assert topology==ptopology
    name=source.stem;names.append(source.name)
    result.append({'model':'GeoTransolver','sample':name,**metrics(true,pred,threshold)})
assert len(set(names))==100 and set(names)=={Path(r['file_path']).name for r in test}
with (out/'per_sample_metrics.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
save('metadata.json',{'global_step':checkpoint.get('global_step'),'val_loss':checkpoint.get('val_loss'),
     'train_manifest':checkpoint.get('train_manifest'),'validation_manifest':checkpoint.get('validation_manifest'),
     'evaluation_metrics':evaluation.get('metrics'),'mape_threshold_pa':threshold,'python':sys.executable,
     'torch':torch.__version__,'run':str(run)})
archive=root/(out.name+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname=out.name)
print('PASS Geo audit export (CPU checkpoint/VTU reads; no training or inference)',flush=True)
print('Download:',archive,flush=True)
print('SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)
PY
