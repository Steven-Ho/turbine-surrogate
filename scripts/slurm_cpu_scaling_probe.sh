#!/bin/bash
#SBATCH --job-name=cpu_scaling
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --get-user-env
#SBATCH --output=logs/cpu_scaling_%j.out
#SBATCH --error=logs/cpu_scaling_%j.err

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
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
probe_root="${TURBINE_PROBE_ROOT:-$PWD/runs/diagnostics}"
mkdir -p "$probe_root"
export TURBINE_PROBE_OUT="$probe_root/cpu_scaling_${SLURM_JOB_ID:?}"
mkdir "$TURBINE_PROBE_OUT"
echo "node=$(hostname) source=$TURBINE_SOURCE_RUN results=$TURBINE_PROBE_OUT"
cat > "$TURBINE_PROBE_OUT/cpu_probe.py" <<'PYTHON_PROBE'
"""Sequential 4/8-core ABBA benchmark on the same allocated GPU and data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import time
import traceback

ROOT=Path(os.environ['TFAI_PROJECT_ROOT'])
SOURCE=Path(os.environ['TURBINE_SOURCE_RUN'])
if not SOURCE.is_absolute(): SOURCE=ROOT/SOURCE
OUT=Path(os.environ['TURBINE_PROBE_OUT'])


def save(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n');temp.replace(path)


def affinity_cpu_topology(cpus):
    result=[]
    for cpu in cpus:
        directory=Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        result.append({'logical_cpu':cpu,'package':(directory/'physical_package_id').read_text().strip() if (directory/'physical_package_id').exists() else None,
                       'core':(directory/'core_id').read_text().strip() if (directory/'core_id').exists() else None})
    return result


def worker(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    cpus=spec['affinity'];os.sched_setaffinity(0,cpus)
    # Thread environment has been set by parent BEFORE these imports.
    import random
    import numpy as np
    import torch
    import yaml
    from tfai.dataset.full_sample_dataset import FullSampleDataset
    from tfai.dataset.data_loader import collate_gcn_batch
    from tfai.factories.model_factory import ModelFactory
    torch.set_num_threads(spec['thread_count'])
    assert torch.cuda.is_available() and torch.cuda.device_count()==1,'Expected one Slurm-assigned GPU'
    config_path=SOURCE/'Transolver_3/configs/stage_20.yaml'
    if not config_path.exists(): config_path=ROOT/'configs/cluster_long/Transolver_3.yaml'
    cfg=yaml.safe_load(config_path.read_text())
    assert cfg['training']['batch_size']==1
    assert not cfg['training']['graph_use_partition']
    processed=SOURCE/'Transolver_3/data/processed'
    table=processed/'effective_train.csv'
    if not table.exists(): table=processed/'train.csv'
    dataset=FullSampleDataset.from_csv(str(table),cfg,str(processed),phase='train')
    count=min(int(os.environ.get('CPU_PROBE_SAMPLES','32')),len(dataset))
    sample_paths=dataset.file_paths[:count]
    def seed():
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    seed()
    if spec['prepare']:
        model=ModelFactory.create_model('Transolver_3','field',cfg)
        torch.save(model.state_dict(),OUT/'initial_weights.pt')
        save(OUT/'sample_manifest.json',{'source':str(SOURCE),'config':str(config_path),
             'files':sample_paths,'ids':dataset.sample_ids[:count],
             'initial_weights_sha256':hashlib.sha256((OUT/'initial_weights.pt').read_bytes()).hexdigest()})
        print('Initial weights and sample order prepared',flush=True)
        return
    manifest=json.loads((OUT/'sample_manifest.json').read_text())
    assert sample_paths==manifest['files'],'Sample order changed'
    assert hashlib.sha256((OUT/'initial_weights.pt').read_bytes()).hexdigest()==manifest['initial_weights_sha256']
    results=[]
    for mode in spec['modes']:
        # Preload to equalize filesystem page-cache warming in ALL conditions.
        # Disk mode still performs torch.load + normalization/collation per step;
        # memory mode removes deserialization but uses the same data and collation.
        samples=[dataset[i] for i in range(count)]
        seed()
        model=ModelFactory.create_model('Transolver_3','field',cfg)
        model.load_state_dict(torch.load(OUT/'initial_weights.pt',map_location='cpu',weights_only=False),strict=True)
        model=model.cuda().train()
        optimizer=torch.optim.Adam(model.parameters(),lr=float(cfg['training']['learning_rate']))
        seed()
        records=[]
        warmup=int(os.environ.get('CPU_PROBE_WARMUP','10'))
        steps=int(os.environ.get('CPU_PROBE_STEPS','100'))
        assert warmup>=1 and steps>=2
        def step(i):
            torch.cuda.synchronize();start=time.perf_counter()
            index=i%count
            payload=dataset[index] if mode=='disk' else samples[index]
            batch,target=collate_gcn_batch([payload]);loaded=time.perf_counter()
            batch=batch.to('cuda');target=target.to('cuda');torch.cuda.synchronize();transferred=time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            prediction=model(batch)
            loss=torch.nn.functional.mse_loss(prediction,target)
            loss.backward();optimizer.step();value=loss.item();torch.cuda.synchronize();end=time.perf_counter()
            assert np.isfinite(value),'Non-finite loss'
            return {'step':i-warmup+1,'sample_index':index,'data_seconds':loaded-start,
                    'transfer_seconds':transferred-loaded,'compute_seconds':end-transferred,
                    'step_seconds':end-start,'loss':value}
        for i in range(warmup): step(i)
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        usage0=resource.getrusage(resource.RUSAGE_SELF);started=time.perf_counter()
        for i in range(warmup,warmup+steps): records.append(step(i))
        torch.cuda.synchronize();wall=time.perf_counter()-started
        usage1=resource.getrusage(resource.RUSAGE_SELF)
        cpu_seconds=(usage1.ru_utime+usage1.ru_stime)-(usage0.ru_utime+usage0.ru_stime)
        result={'case':spec['case'],'repeat':spec['repeat'],'cpu_count':spec['cpu_count'],'thread_count':spec['thread_count'],'mode':mode,
                'steps':steps,'warmup':warmup,'sample_count':count,'affinity':sorted(os.sched_getaffinity(0)),
                'physical_cpu_topology':affinity_cpu_topology(cpus),'torch_threads':torch.get_num_threads(),
                'torch_interop_threads':torch.get_num_interop_threads(),'hostname':platform.node(),
                'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'python':platform.python_version(),
                'wall_seconds':wall,'mean_step_seconds':wall/steps,
                'median_step_seconds':statistics.median(r['step_seconds'] for r in records),
                'mean_data_seconds':statistics.mean(r['data_seconds'] for r in records),
                'mean_transfer_seconds':statistics.mean(r['transfer_seconds'] for r in records),
                'mean_compute_seconds':statistics.mean(r['compute_seconds'] for r in records),
                'effective_cpu_cores':cpu_seconds/wall,'process_cpu_seconds':cpu_seconds,
                'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'final_loss':records[-1]['loss'],
                'initial_weights_sha256':manifest['initial_weights_sha256']}
        results.append(result)
        with (OUT/f'{spec["case"]}_{mode}_steps.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        save(OUT/f'{spec["case"]}_results.json',results)
        print(json.dumps(result),flush=True)
        del model,optimizer,samples
        torch.cuda.empty_cache()


def parent():
    counts=[int(n) for n in os.environ.get('CPU_PROBE_COUNTS','4,8').split(',')]
    assert counts and all(n>0 for n in counts) and len(set(counts))==len(counts)
    allowed=sorted(os.sched_getaffinity(0))
    assert max(counts)<=len(allowed),f'Need >= {max(counts)} allocated logical CPUs, got {len(allowed)}. Increase sbatch --cpus-per-task.'
    fixed_threads=int(os.environ.get('CPU_PROBE_THREADS','4'))
    assert fixed_threads>=0 and (fixed_threads==0 or fixed_threads<=min(counts)), 'Fixed threads must fit the smallest CPU count; use 0 for proportional threads'
    repeats=int(os.environ.get('CPU_PROBE_REPEATS','2'))
    assert repeats>=1
    save(OUT/'allocation.json',{'allowed_cpu_ids':allowed,'topology':affinity_cpu_topology(allowed),
         'counts':counts,'repeats':repeats,'fixed_threads':fixed_threads,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
         'SLURM_STEP_GPUS':os.environ.get('SLURM_STEP_GPUS'),'source':str(SOURCE)})
    def launch(spec):
        path=OUT/f'{spec["case"]}_spec.json';save(path,spec)
        env=dict(os.environ,OMP_NUM_THREADS=str(spec['thread_count']),MKL_NUM_THREADS=str(spec['thread_count']),
                 OPENBLAS_NUM_THREADS=str(spec['thread_count']),NUMEXPR_NUM_THREADS=str(spec['thread_count']))
        print(f'Starting {spec["case"]}: {spec["cpu_count"]} logical CPUs',flush=True)
        with (OUT/f'{spec["case"]}.log').open('w') as f:
            result=subprocess.run([sys.executable,__file__,'--worker',str(path)],env=env,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{spec["case"]} failed: see {OUT / (spec["case"] + ".log")}')
    launch({'case':'prepare','prepare':True,'cpu_count':max(counts),'thread_count':fixed_threads or max(counts),'affinity':allowed[:max(counts)]})
    all_results=[];case_index=0
    for repeat in range(repeats):
        order=counts if repeat%2==0 else list(reversed(counts))
        for n in order:
            name=f'repeat{repeat+1}_cpu{n}'
            launch({'case':name,'prepare':False,'repeat':repeat+1,'cpu_count':n,'thread_count':fixed_threads or n,'affinity':allowed[:n],
                    'modes':['disk','memory'] if case_index%2==0 else ['memory','disk']})
            all_results+=json.loads((OUT/f'{name}_results.json').read_text());case_index+=1
            save(OUT/'all_results.json',all_results)
    summary=[]
    for mode in ('disk','memory'):
        base_values=[r['mean_step_seconds'] for r in all_results if r['cpu_count']==counts[0] and r['mode']==mode]
        baseline=statistics.median(base_values)
        for n in counts:
            selected=[r for r in all_results if r['cpu_count']==n and r['mode']==mode]
            median=statistics.median(r['mean_step_seconds'] for r in selected)
            summary.append({'cpu_count':n,'thread_count':fixed_threads or n,'mode':mode,'repeats':len(selected),'median_mean_step_seconds':median,
                            'min_mean_step_seconds':min(r['mean_step_seconds'] for r in selected),
                            'max_mean_step_seconds':max(r['mean_step_seconds'] for r in selected),
                            'speedup_vs_first_count':baseline/median,
                            'mean_data_seconds':statistics.mean(r['mean_data_seconds'] for r in selected),
                            'mean_transfer_seconds':statistics.mean(r['mean_transfer_seconds'] for r in selected),
                            'mean_compute_seconds':statistics.mean(r['mean_compute_seconds'] for r in selected),
                            'mean_effective_cpu_cores':statistics.mean(r['effective_cpu_cores'] for r in selected)})
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    lines=['# Transolver_3 CPU scaling probe','',
           'Same allocated GPU, real full-grid samples, shared initial weights, seed=42, Adam, original LR and batch_size=1.',
           'Default order is 4 -> 8 -> 8 -> 4 logical CPUs. Threads stay at 4 to isolate affinity/allocation capacity, matching the original training settings.',
           'Set CPU_PROBE_THREADS=0 to scale threads with CPU count instead. This changes both affinity and numerical thread caps.',
           'Each condition runs in a fresh process; affinity and thread limits are set before torch imports.',
           'Each case warms up and measures the same sample sequence. No source checkpoint/cache is modified.',
           '', '| CPU count | threads | mode | sec/step | speedup vs first CPU count | effective CPU cores |',
           '|---:|---:|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["cpu_count"]} | {r["thread_count"]} | {r["mode"]} | {r["median_mean_step_seconds"]:.6f} | {r["speedup_vs_first_count"]:.3f} | {r["mean_effective_cpu_cores"]:.2f} |')
    lines+=['','disk: serial processed-sample loading/deserialization/collation each step, after equal page-cache warming.',
            'memory: preloaded identical CPU samples, with identical collation and CPU->GPU transfer.',
            'Timing includes synchronization instrumentation; it is not the full training/validation/checkpoint pipeline.',
            'The disk path reproduces num_workers=0 loading semantics; it directly calls the same dataset and collator.',
            'If more CPUs consistently reduce both modes, CPU thread capacity/launch overhead may matter.',
            'If only disk is slow while memory is fast, sample loading/deserialization is a more likely bottleneck.',
            'If neither mode improves, additional CPUs alone are unlikely to explain the observed training slowdown.',
            'Low effective CPU usage does not rule out a single-thread bottleneck; more cores cannot accelerate serial work.',
            'GPU/backend kernels, NUMA locality, clock variation, concurrent I/O and measurement order can still affect results.',
            'Read allocation.json and each case result for logical/physical CPU topology. These are not necessarily physical-core counts.','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    save(OUT/'status.json',{'state':'complete'})
    print('COMPLETE:',OUT,flush=True)


if __name__=='__main__':
    try:
        if len(sys.argv)>1 and sys.argv[1]=='--worker':worker(sys.argv[2])
        else:parent()
    except Exception:
        if len(sys.argv)==1:save(OUT/'status.json',{'state':'failed','traceback':traceback.format_exc()})
        raise

PYTHON_PROBE
exec srun --ntasks=1 --gres=gpu:1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}" python "$TURBINE_PROBE_OUT/cpu_probe.py"
