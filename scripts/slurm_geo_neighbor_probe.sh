#!/bin/bash
#SBATCH --job-name=geo_neighbors
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --get-user-env
#SBATCH --output=logs/geo_neighbors_%j.out
#SBATCH --error=logs/geo_neighbors_%j.err

# Submit from the project root AFTER: mkdir -p logs
# export TURBINE_SOURCE_RUN=runs/metax_long_20261111_100428_4166078
set -eo pipefail
module purge
module load conda-py31010/3.10.10
source "${TURBINE_VENV:-$HOME/venvs/turbine-metax}/bin/activate"
cd "${SLURM_SUBMIT_DIR:?Submit using sbatch from the project root}"
source "${TURBINE_OPENGL_RUNTIME:-$PWD/turbine-opengl-jammy}/load.sh"
: "${TURBINE_SOURCE_RUN:?Export TURBINE_SOURCE_RUN to the existing seven-model result directory}"
export TFAI_PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONUTF8=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
probe_root="${TURBINE_PROBE_ROOT:-$PWD/runs/diagnostics}"
mkdir -p "$probe_root"
export TURBINE_PROBE_OUT="$probe_root/geo_neighbors_${SLURM_JOB_ID:?}"
mkdir "$TURBINE_PROBE_OUT"
echo "node=$(hostname) source=$TURBINE_SOURCE_RUN results=$TURBINE_PROBE_OUT"
cat > "$TURBINE_PROBE_OUT/geo_probe.py" <<'PYTHON_PROBE'
import csv
import json
import os
from pathlib import Path
import platform
import time
import traceback

import torch
import yaml

ROOT = Path(os.environ['TFAI_PROJECT_ROOT'])
SOURCE = Path(os.environ['TURBINE_SOURCE_RUN'])
if not SOURCE.is_absolute(): SOURCE = ROOT / SOURCE
OUT = Path(os.environ['TURBINE_PROBE_OUT'])


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def sync():
    torch.cuda.synchronize()


def timed(call):
    sync();start = time.perf_counter();result = call();sync()
    return result, time.perf_counter() - start


def main():
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one Slurm-assigned GPU'
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
    from tfai.dataset.full_sample_dataset import FullSampleDataset
    from tfai.dataset.data_loader import collate_gcn_batch
    from tfai.factories.model_factory import ModelFactory
    from tfai.models._vendor.geotransolver2.ball_query import BQWarp
    model_dir = SOURCE / 'GeoTransolver'
    config_path = model_dir / 'configs/stage_20.yaml'
    if not config_path.exists(): config_path = ROOT / 'configs/cluster_long/GeoTransolver.yaml'
    cfg = yaml.safe_load(config_path.read_text())
    cfg['model']['geotransolver_backend'] = 'portable_torch'
    processed = model_dir / 'data/processed'
    table = processed / 'effective_train.csv'
    if not table.exists(): table = processed / 'train.csv'
    dataset = FullSampleDataset.from_csv(str(table), cfg, str(processed), phase='train')
    count = min(int(os.environ.get('GEO_PROBE_SAMPLES', '3')), len(dataset))
    assert count >= 2, 'Need at least two distinct samples'
    chunk = int(os.environ.get('GEO_PROBE_CHUNK', '256'))
    payloads = [dataset[i] for i in range(count)]
    coords = [p[0].x[:, :3].contiguous() for p in payloads]
    differences = [{'sample_index': i, 'file': dataset.file_paths[i],
                    'equal_to_first': torch.equal(coords[0], c),
                    'max_abs_difference': float((coords[0] - c).abs().max()) if c.shape == coords[0].shape else None}
                   for i,c in enumerate(coords)]
    save('environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
         'gpu': torch.cuda.get_device_name(0), 'hostname': platform.node(),
         'affinity': sorted(os.sched_getaffinity(0)), 'source': str(SOURCE),
         'config': str(config_path), 'nodes': [len(c) for c in coords], 'chunk': chunk,
         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'SLURM_STEP_GPUS': os.environ.get('SLURM_STEP_GPUS')})
    save('geometry_differences.json', differences)
    print('Geometry differences:', json.dumps(differences), flush=True)
    gpu_coords = [c.cuda().unsqueeze(0) for c in coords]
    sequence = [('A_cold', 0), ('A_same_value_clone', 0)]
    sequence += [(f'different_sample_{i}', i) for i in range(1,count)]
    sequence += [('return_to_A', 0)]
    rows = []
    for radius,k in ((.05,8),(.25,32)):
        query = BQWarp(radius, k, query_chunk_size=chunk).cuda()
        for name,i in sequence:
            x = gpu_coords[i].clone() if name == 'A_same_value_clone' else gpu_coords[i]
            hits,misses = query.cache_hits,query.cache_misses
            torch.cuda.reset_peak_memory_stats()
            print(f'Query radius={radius} case={name} nodes={x.shape[1]}', flush=True)
            result,seconds = timed(lambda: query(x,x))
            assert torch.isfinite(result[1]).all()
            row = {'radius': radius, 'neighbors': k, 'case': name, 'sample_index': i,
                   'seconds': seconds, 'cache_hits_delta': query.cache_hits-hits,
                   'cache_misses_delta': query.cache_misses-misses,
                   'peak_gpu_bytes': torch.cuda.max_memory_allocated()}
            rows.append(row);save('neighbor_query_results.json',rows)
            print(json.dumps(row),flush=True)
        # One representative block, split into cdist and topk to locate the kernel cost.
        x = gpu_coords[0][0]
        distances,cdist_seconds = timed(lambda: torch.cdist(x,x[:chunk],p=2.,compute_mode='donot_use_mm_for_euclid_dist'))
        _,topk_seconds = timed(lambda: torch.topk(distances,k=min(k,len(x)),dim=0,largest=False))
        save(f'kernel_block_radius_{radius}.json',{'source_points':len(x),'query_points':min(chunk,len(x)),
             'cdist_seconds':cdist_seconds,'topk_seconds':topk_seconds,
             'note':'Single block timing; not extrapolated to the full query.'})
        del query,distances,result
    torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    model = ModelFactory.create_model('GeoTransolver','field',cfg).cuda().train()
    query_events = []
    current = {'case': ''}
    # Instrument the actual model's query path. Synchronization adds profiling overhead.
    for module_name,module in model.named_modules():
        if not isinstance(module,BQWarp): continue
        module.query_chunk_size = chunk
        original = module._indices
        def wrapped(points,queries,original=original,module=module,module_name=module_name):
            hits,misses = module.cache_hits,module.cache_misses
            result,seconds = timed(lambda: original(points,queries))
            query_events.append({'case':current['case'],'module':module_name,'seconds':seconds,
                                 'cache_hits_delta':module.cache_hits-hits,'cache_misses_delta':module.cache_misses-misses})
            return result
        module._indices = wrapped
    model_rows = []
    for name,i in sequence:
        current['case'] = name
        start_index = len(query_events)
        batch,target = collate_gcn_batch([payloads[i]])
        (batch,target),transfer_seconds = timed(lambda:(batch.to('cuda'),target.to('cuda')))
        model.zero_grad(set_to_none=True)
        print('Model forward/backward:',name,flush=True)
        sync();start = time.perf_counter()
        prediction = model(batch)
        loss = torch.nn.functional.mse_loss(prediction,target)
        loss.backward();sync()
        elapsed = time.perf_counter()-start
        events = query_events[start_index:]
        assert torch.isfinite(loss).item()
        row = {'case':name,'sample_index':i,'transfer_seconds':transfer_seconds,
               'forward_backward_seconds':elapsed,'query_indices_seconds':sum(e['seconds'] for e in events),
               'query_calls':len(events),'cache_hits':sum(e['cache_hits_delta'] for e in events),
               'cache_misses':sum(e['cache_misses_delta'] for e in events),'loss':loss.item()}
        model_rows.append(row);save('model_results.json',model_rows);save('query_events.json',query_events)
        print(json.dumps(row),flush=True)
    for filename,data in [('neighbor_query_results.csv',rows),('model_results.csv',model_rows),('query_events.csv',query_events)]:
        with (OUT/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    lines=['# GeoTransolver geometry probe','',
           'Compare A_cold / A_same_value_clone / different_sample_* / return_to_A in neighbor_query_results.csv.',
           'A repeated value should hit; distinct coordinates and returning to an evicted A should miss.',
           'model_results.csv measures the full real model forward/backward and the portion spent querying indices.',
           'If different samples repeatedly miss and query time dominates, quadratic query recomputation is a confirmed bottleneck.',
           'If cache hits are fast but the model remains slow, inspect other model kernels/backward instead.',
           'Timing is synchronized diagnostic timing with instrumentation overhead, not a native training speed score.',
           'The source training cache/config/checkpoints were read-only. No optimizer update was performed.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    save('status.json',{'state':'complete'})
    print('COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    try: main()
    except Exception:
        save('status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise

PYTHON_PROBE
exec srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" python "$TURBINE_PROBE_OUT/geo_probe.py"
