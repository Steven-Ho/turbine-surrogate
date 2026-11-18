"""One Slurm array task: reproduce 20 -> 200 -> optional 400 epoch training."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['POD', 'UNet', 'GCN', 'Graph_UNet', 'Graph_Transformer', 'Transolver_3', 'GeoTransolver']


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def duration(text):
    matches = re.findall(r'模型训练耗时: ([^\n]+)', text)
    if not matches:
        return None
    scales = {'小时': 3600, '时': 3600, '分': 60, '秒': 1}
    return sum(float(v) * scales[u] for v, u in re.findall(r'([0-9.]+)\s*(小时|时|分|秒)', matches[-1]))


def should_extend(history):
    units = history.get('val_units', [])
    losses = history.get('val_loss', [])
    if not units or units[-1] < 200 * 800:
        return False
    before = [loss for step, loss in zip(units, losses) if step <= 160 * 800]
    recent = [loss for step, loss in zip(units, losses) if step > 160 * 800]
    return bool(before and recent and min(recent) < .98 * min(before))


def make_tables(run):
    source = ROOT / 'exp_rotor37_GCN/data/raw/samples.csv'
    with source.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1000
    paths = [(source.parent / r['file_path']).resolve() for r in rows]
    assert len(set(paths)) == 1000 and all(p.is_file() for p in paths), 'Incomplete Rotor37 data'
    for row, path in zip(rows, paths):
        row['file_path'] = str(path)
    random.Random(42).shuffle(rows)
    directory = run / 'input'
    directory.mkdir()
    for name, subset in [('samples.csv', rows[100:]), ('test.csv', rows[:100])]:
        with (directory / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(subset)
    write_json(run / 'data_split.json', {
        'seed': 42, 'pool': [Path(r['file_path']).name for r in rows[100:]],
        'test': [Path(r['file_path']).name for r in rows[:100]],
        'source_table_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    })


def export_history(torch, checkpoint, path):
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    fields = ['epoch', 'numEpochs', 'totalStep', 'loss', 'valLoss', 'lr', 'timestamp']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(cp.get('progress_history', []))
    return cp


def build_stage_config(frozen, baseline, run, target):
    cfg = yaml.safe_load(yaml.safe_dump(frozen))
    cfg.pop('_runtime', None)
    cfg['general'].update(experiment=str(run), train_samples=str(run / 'input/samples.csv'),
                          train_processed_dir=str(run / 'data/processed'), device='cuda:0')
    cfg['model']['checkpoint_path'] = str(run / 'train/best_model.pt')
    train = cfg['training']
    if target > 20:
        train.update(json.loads(json.dumps(baseline['long_protocol'])))
    train.update(epochs=target, max_steps=None, validate_every=None, validation_max_batches=None,
                 auto_evaluate_after_train=False, require_preprocess_cache=(target > 20))
    train['resume'] = {'enabled': target > 20, 'path': str(run / 'train/last_checkpoint.pt')}
    cfg['hydra'] = {'run': {'dir': str(run / f'hydra_{target}')}, 'output_subdir': None}
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--plan-only', action='store_true', help='Resolve stage configs without importing torch or starting training')
    args = parser.parse_args()
    import yaml
    baseline = json.loads((ROOT / 'configs/cluster_long/local_baseline.json').read_text())
    frozen = yaml.safe_load((ROOT / f'configs/cluster_long/{args.model}.yaml').read_text())
    if args.model == 'GeoTransolver':
        frozen['model']['geotransolver_backend'] = 'portable_torch'
    parent = (ROOT / args.run_root).resolve()
    run = parent / args.model
    if args.plan_only:
        print(json.dumps({'model': args.model, 'run': str(run), 'stages': [20, 200, 400],
                          'base_training': frozen['training'], 'long_training': baseline['long_protocol'],
                          'extension': '200 epochs reached AND >2% late validation improvement',
                          'local_long_baseline_available': bool(baseline['models'][args.model]['long_stages'])}, indent=2))
        return
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'status.json').exists():
        raise RuntimeError('Run already exists. Use a new TURBINE_LONG_RUN_ROOT to avoid overwriting results.')
    status = {'model': args.model, 'state': 'preflight', 'start_time': time.time(), 'stages': {},
              'protocol': '20 constant-LR epochs, continue to 200 with plateau/early stop, extend to 400 if warranted',
              'run_directory': str(run), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_array_job_id': os.environ.get('SLURM_ARRAY_JOB_ID'),
              'local_long_baseline_available': bool(baseline['models'][args.model]['long_stages'])}
    write_json(run / 'status.json', status)
    try:
        import torch
        import vtk
        diagnostic = {
            'hostname': platform.node(), 'python': sys.executable,
            'torch': torch.__version__, 'torch_file': torch.__file__,
            'gpu_available': torch.cuda.is_available(), 'gpu_count': torch.cuda.device_count(),
            'environment': {k: os.environ.get(k) for k in (
                'CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS',
                'SLURM_LOCALID', 'SLURM_PROCID', 'MACA_VISIBLE_DEVICES',
                'HT_VISIBLE_DEVICES', 'MARS_VISIBLE_DEVICES', 'LD_LIBRARY_PATH')},
        }
        write_json(run / 'gpu_preflight.json', diagnostic)
        print('GPU preflight: ' + json.dumps(diagnostic, ensure_ascii=False), flush=True)
        if shutil.which('ht-smi'):
            try:
                device_log = subprocess.run(['ht-smi'], capture_output=True, text=True, timeout=20)
                (run / 'ht-smi.txt').write_text(device_log.stdout + device_log.stderr)
            except Exception as exc:
                (run / 'ht-smi.txt').write_text(str(exc))
        assert diagnostic['gpu_available'], 'No GPU available; inspect gpu_preflight.json and ht-smi.txt for this node/job step' 
        assert torch.cuda.device_count() == 1, 'Expected exactly one Slurm-assigned visible GPU. Ask the site to configure CUDA_VISIBLE_DEVICES/SLURM_JOB_GPUS; do not run multiple tasks on an unbound GPU 0.'
        environment = {'python': sys.version, 'python_executable': sys.executable, 'torch': torch.__version__,
                       'torch_file': torch.__file__, 'gpu': torch.cuda.get_device_name(0),
                       'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                       'vtk': vtk.vtkVersion.GetVTKVersion(), 'platform': platform.platform(),
                       'hostname': platform.node(), 'model_backend': frozen['model'].get('geotransolver_backend', 'default'), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
                       'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}}
        write_json(run / 'environment.json', environment)
        shutil.copy2(ROOT / 'configs/cluster_long/local_baseline.json', run / 'local_baseline.json')
        # Early, small graph forward/backward exposes missing PyG sparse extensions or Warp support.
        if args.model in {'Graph_UNet', 'GeoTransolver'}:
            from tfai.factories.model_factory import ModelFactory
            from torch_geometric.data import Data
            model = ModelFactory.create_model(args.model, 'field', frozen).cuda()
            n = 32
            x = torch.randn(n, frozen['model']['in_channels'], device='cuda')
            if args.model == 'GeoTransolver':
                x[:, 3:] = x[:1, 3:].clone()
            src = torch.arange(n, device='cuda')
            edge = torch.stack([src, (src + 1) % n])
            data = Data(x=x, edge_index=torch.cat([edge, edge.flip(0)], dim=1))
            model(data).square().mean().backward()
            torch.cuda.synchronize()
            del model, data
            torch.cuda.empty_cache()
        make_tables(run)
        configs = run / 'configs'
        configs.mkdir()
        for target in (20, 200, 400):
            if target == 400:
                last = torch.load(run / 'train/last_checkpoint.pt', map_location='cpu', weights_only=False)
                status['extension'] = should_extend(last['history'])
                if not status['extension']:
                    status['extension_reason'] = 'Did not reach 200 epochs, or late validation improvement <=2%'
                    break
            cfg = build_stage_config(frozen, baseline, run, target)
            (configs / f'stage_{target}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
            command = [sys.executable, 'scripts/train.py', '--config-path', str(configs), '--config-name', f'stage_{target}']
            status['state'] = f'training_{target}'
            stage = {'command': command, 'start_time': time.time()}
            status['stages'][str(target)] = stage
            write_json(run / 'status.json', status)
            print(f'{args.model}: starting stage {target}; log={run / f"train_{target}_console.log"}', flush=True)
            started = time.perf_counter()
            with (run / f'train_{target}_console.log').open('w') as output:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            stage.update(pipeline_train_seconds=time.perf_counter() - started, exit_code=result.returncode)
            if (run / 'train/train.log').exists():
                shutil.copy2(run / 'train/train.log', run / f'train_{target}.log')
                stage['training_seconds'] = duration((run / f'train_{target}.log').read_text(errors='replace'))
            if result.returncode:
                raise RuntimeError(f'Train stage {target} failed; see train_{target}_console.log')
            last = export_history(torch, run / 'train/last_checkpoint.pt', run / 'training_history.csv')
            completed = last['global_step']
            stage.update(completed_steps=completed, completed_epochs=completed / 800,
                         final_lr=last['optimizer_state_dict']['param_groups'][0]['lr'])
            assert last['config']['training']['learning_rate'] == 1e-4
            assert last['config']['training']['batch_size'] == 1
            for split, expected in [('train', 800), ('valid', 100)]:
                with (run / f'data/processed/{split}.csv').open() as f:
                    rows = list(csv.DictReader(f))
                    assert len(rows) == expected
                    actual_names = sorted(Path(row['file_path']).name for row in rows)
                    assert actual_names == baseline['models'][args.model]['split_sample_names'][split], f'{split} split differs from local baseline'
            assert torch.isfinite(torch.tensor(last['history']['val_loss'])).all()
            milestone = run / 'milestones' / str(target)
            milestone.mkdir(parents=True)
            for name in ('best_model.pt', 'last_checkpoint.pt'):
                shutil.copy2(run / 'train' / name, milestone / name)
            stage['end_time'] = time.time()
            print(f'{args.model}: stage {target} finished at epoch {completed / 800}; training_seconds={stage.get("training_seconds")}', flush=True)
            write_json(run / 'status.json', status)
            if completed < target * 800:
                status['early_stopped'] = True
                break
        status['state'] = 'evaluating'
        write_json(run / 'status.json', status)
        command = [sys.executable, 'scripts/evaluate.py', f'evaluate.experiment={run}',
                   f'evaluate.evaluate_samples={run / "input/test.csv"}', 'evaluate.strict_writeback=true',
                   'evaluate.plot_scatter=false', f'hydra.run.dir={run / "hydra_evaluate"}']
        started = time.perf_counter()
        with (run / 'evaluate_console.log').open('w') as output:
            result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        status['pipeline_test_seconds'] = time.perf_counter() - started
        if result.returncode:
            raise RuntimeError('Evaluation failed; see evaluate_console.log')
        status['state'] = 'complete'
        print(f'{args.model}: complete; results={run}', flush=True)
    except Exception as exc:
        status.update(state='failed', error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
        raise
    finally:
        status['end_time'] = time.time()
        status['task_seconds'] = status['end_time'] - status['start_time']
        write_json(run / 'status.json', status)


if __name__ == '__main__':
    main()
