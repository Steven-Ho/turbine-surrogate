#!/bin/bash
#SBATCH --job-name=geo_opt_verify
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --get-user-env
#SBATCH --output=logs/geo_opt_verify_%j.out
#SBATCH --error=logs/geo_opt_verify_%j.err

set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit from project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
export TFAI_PROJECT_ROOT="$PWD" PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TURBINE_GEO_QUERY_BACKEND=kdtree
export TURBINE_GEO_CACHE_DIR="${TURBINE_GEO_CACHE_DIR:-$PWD/runs/geo_neighbor_cache_kdtree_v1}"
: "${TURBINE_SOURCE_RUN:?Export the existing seven-model result directory}"
mkdir -p runs/diagnostics
export TURBINE_GEO_VERIFY_OUT="$PWD/runs/diagnostics/geo_opt_verify_${SLURM_JOB_ID}"
mkdir "$TURBINE_GEO_VERIFY_OUT"
cat > "$TURBINE_GEO_VERIFY_OUT/verify.py" <<'PYTHON_VERIFY'
import csv,json,os,time,traceback
from pathlib import Path
import torch,yaml
from tfai.dataset.full_sample_dataset import FullSampleDataset
from tfai.dataset.data_loader import collate_gcn_batch
from tfai.factories.model_factory import ModelFactory
from tfai.models._vendor.geotransolver2.ball_query import BQWarp

ROOT=Path(os.environ['TFAI_PROJECT_ROOT']);OUT=Path(os.environ['TURBINE_GEO_VERIFY_OUT'])

def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def timed(call):
    torch.cuda.synchronize();start=time.perf_counter();v=call();torch.cuda.synchronize();return v,time.perf_counter()-start


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    source=Path(os.environ['TURBINE_SOURCE_RUN'])
    if not source.is_absolute():source=ROOT/source
    directory=source/'GeoTransolver/data/processed'
    cfg_path=source/'GeoTransolver/configs/stage_20.yaml'
    if not cfg_path.exists():cfg_path=ROOT/'configs/cluster_long/GeoTransolver.yaml'
    cfg=yaml.safe_load(cfg_path.read_text());cfg['model']['geotransolver_backend']='portable_torch'
    table=directory/'effective_train.csv'
    if not table.exists():table=directory/'train.csv'
    ds=FullSampleDataset.from_csv(str(table),cfg,str(directory),phase='train')
    samples=[ds[i] for i in range(min(2,len(ds)))];assert len(samples)>=2
    rows=[]
    for radius,k in ((.05,8),(.25,32)):
        ref=BQWarp(radius,k,backend='torch').cuda();fast=BQWarp(radius,k,backend='kdtree').cuda()
        for i,(graph,_) in enumerate(samples):
            points=graph.x[:,:3].contiguous().cuda().unsqueeze(0)
            print(f'Checking all nodes: radius={radius} sample={i}',flush=True)
            expected,old_seconds=timed(lambda:ref(points,points))
            actual,new_seconds=timed(lambda:fast(points,points))
            mismatches=int((expected[0]!=actual[0]).sum().item())
            row={'radius':radius,'sample':i,'index_mismatches':mismatches,'reference_seconds':old_seconds,
                 'new_cold_or_cached_seconds':new_seconds,'tree_builds':fast.tree_builds,
                 'reference_fallback_chunks':fast.reference_chunks,'ambiguity_rows':fast.ambiguity_rows}
            rows.append(row);save('query_validation.json',rows)
            assert mismatches==0, 'Neighbor indices differ from the legacy query; do not start training'
            torch.testing.assert_close(expected[1],actual[1],rtol=0,atol=0)
        points=samples[0][0].x[:,:3].contiguous().cuda().unsqueeze(0)
        _,seconds=timed(lambda:fast(points,points))
        fresh=BQWarp(radius,k,backend='kdtree').cuda()
        _,disk_seconds=timed(lambda:fresh(points,points))
        assert fresh.disk_cache_hits==1 and fresh.tree_builds==0
        save(f'cache_reuse_{radius}.json',{'return_to_A_seconds':seconds,'memory_hits':fast.memory_cache_hits,
             'fresh_instance_seconds':disk_seconds,'disk_hits':fresh.disk_cache_hits})
    models=[]
    for backend in ('torch','kdtree'):
        os.environ['TURBINE_GEO_QUERY_BACKEND']=backend
        torch.manual_seed(42);model=ModelFactory.create_model('GeoTransolver','field',cfg).cuda().train()
        models.append(model)
    models[1].load_state_dict(models[0].state_dict(),strict=True)
    x,y=collate_gcn_batch([samples[0]]);x=x.to('cuda');y=y.cuda()
    values=[];times=[]
    for backend,model in zip(('torch','kdtree'),models):
        def step():
            model.zero_grad(set_to_none=True);p=model(x);loss=torch.nn.functional.mse_loss(p,y);loss.backward();return p.detach()
        value,seconds=timed(step);values.append(value);times.append(seconds)
        print(f'Full-grid {backend} forward/backward seconds={seconds:.6f}',flush=True)
    torch.testing.assert_close(values[0],values[1],rtol=3e-5,atol=3e-6)
    max_grad_diff=0.
    for (name,p),(other,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
        assert name==other
        if p.grad is None or q.grad is None:assert p.grad is None and q.grad is None
        else:
            torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=3e-6)
            max_grad_diff=max(max_grad_diff,float((p.grad-q.grad).abs().max().item()))
    save('model_validation.json',{'state_dict_keys_equal':True,'outputs_and_parameter_gradients_close':True,
         'max_output_abs_diff':float((values[0]-values[1]).abs().max().item()),'max_parameter_grad_abs_diff':max_grad_diff,
         'reference_forward_backward_seconds':times[0],'optimized_cached_forward_backward_seconds':times[1],
         'note':'Optimized disk cache warmed by equality checks; cold construction times are recorded separately.'})
    save('status.json',{'state':'complete','all_checks_passed':True})
    print('ALL CHECKS PASSED',flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        save('status.json',{'state':'failed','traceback':traceback.format_exc()});raise

PYTHON_VERIFY
exec srun --ntasks=1 --gres=gpu:1 python "$TURBINE_GEO_VERIFY_OUT/verify.py"
