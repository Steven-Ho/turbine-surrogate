#!/usr/bin/env python3
"""GPU inventory and reproducible PyTorch microbenchmarks (CUDA / MetaX MACA)."""
import argparse
import datetime as dt
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def command(argv):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {'command': argv, 'returncode': p.returncode,
                'stdout': p.stdout, 'stderr': p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': argv, 'error': str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New output directory (must not exist)')
    parser.add_argument('--inventory-only', action='store_true', help='Skip load tests')
    parser.add_argument('--buffer-mib', type=positive, default=256, help='Bytes per copy buffer, MiB')
    parser.add_argument('--matrix-size', type=positive, default=4096)
    parser.add_argument('--repeats', type=positive, default=5)
    parser.add_argument('--iterations', type=positive, default=20)
    parser.add_argument('--devices', help='Comma-separated PyTorch logical GPU IDs; default all visible')
    args = parser.parse_args()
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = args.output or Path('runs/diagnostics') / ('gpu_' + platform.node() + '_' + stamp)
    out.mkdir(parents=True, exist_ok=False)
    report = {'started_utc': stamp, 'host': platform.node(), 'python': sys.version,
              'executable': sys.executable, 'platform': platform.platform(),
              'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'environment': {k: os.environ.get(k) for k in (
                  'CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'MACA_VISIBLE_DEVICES',
                  'SLURM_JOB_ID', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURM_CPUS_PER_TASK')},
              'gpus': [], 'errors': []}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')

    smi = [name for name in ('ht-smi', 'nvidia-smi') if shutil.which(name)]
    inventory = [command([name]) for name in smi]
    if 'nvidia-smi' in smi:
        inventory += [command(['nvidia-smi', '-q']), command(['nvidia-smi', 'topo', '-m'])]
    # Preserve vendor output without assuming that ht-smi implements NVIDIA flags.
    if 'ht-smi' in smi:
        inventory.append(command(['ht-smi', '--help']))
    (out / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    stop = threading.Event()
    phase = {'name': 'initializing'}

    def sample():
        with (out / 'telemetry.jsonl').open('w') as f:
            while not stop.is_set():
                for name in smi:
                    label = phase['name']
                    result = command([name, '-q'] if name == 'nvidia-smi' else [name])
                    f.write(json.dumps({'utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                                        'phase': label, **result}) + '\n')
                    f.flush()
                stop.wait(1)

    monitor = threading.Thread(target=sample, daemon=True)
    try:
        import torch
        report['torch'] = {'version': torch.__version__, 'path': torch.__file__,
                           'cuda': torch.version.cuda, 'hip': getattr(torch.version, 'hip', None)}
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU visible to PyTorch; check job allocation and vendor environment')
        count = torch.cuda.device_count()
        devices = list(range(count)) if args.devices is None else [int(x) for x in args.devices.split(',')]
        if not devices or len(set(devices)) != len(devices) or any(i < 0 or i >= count for i in devices):
            raise ValueError('devices must be unique visible logical GPU IDs in [0, %d)' % count)
        torch.set_num_threads(max(1, int(os.environ.get('OMP_NUM_THREADS', '1'))))
        torch.backends.cuda.matmul.allow_tf32 = False
        monitor.start()

        def measure(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(args.iterations):
                    fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / args.iterations)
            median = statistics.median(samples)
            if median <= 0:
                raise RuntimeError('GPU event timer returned non-positive elapsed time')
            return {'median_ms': median, 'min_ms': min(samples), 'max_ms': max(samples),
                    'samples_ms': samples}

        for index in devices:
            torch.cuda.set_device(index)
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info()
            gpu = {'logical_id': index, 'name': props.name, 'properties_raw': str(props),
                   'total_memory_bytes': total, 'free_memory_bytes_before': free,
                   'benchmarks': []}
            for key in ('uuid', 'pci_bus_id', 'major', 'minor', 'multi_processor_count',
                        'clock_rate', 'memory_clock_rate', 'memory_bus_width', 'L2_cache_size'):
                value = getattr(props, key, None)
                if value is not None:
                    gpu[key] = value if isinstance(value, (int, float, str)) else str(value)
            report['gpus'].append(gpu)
            print('GPU %d: %s; total %.2f GiB, free %.2f GiB' % (index, props.name, total/2**30, free/2**30), flush=True)
            save()
            if args.inventory_only:
                continue

            def run(name, fn):
                phase['name'] = 'gpu_%d/%s' % (index, name)
                print('  ' + phase['name'], flush=True)
                try:
                    row = {'name': name, 'status': 'ok', **fn()}
                except Exception as exc:
                    row = {'name': name, 'status': 'error', 'error': '%s: %s' % (type(exc).__name__, exc)}
                    report['errors'].append({'gpu': index, **row})
                gpu['benchmarks'].append(row)
                print('  ' + json.dumps(row), flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                save()

            def copy_test(direction):
                available, _ = torch.cuda.mem_get_info()
                size = min(args.buffer_mib * 2**20, int(available * .10)) // 4 * 4
                if size < 2**20:
                    raise RuntimeError('Insufficient free memory for copy benchmark')
                n = size // 4
                if direction == 'device_copy':
                    src = torch.ones(n, device='cuda', dtype=torch.float32)
                    dst = torch.empty_like(src)
                    traffic = 2 * size  # One read plus one write to GPU memory.
                elif direction == 'host_to_device':
                    src = torch.ones(n, pin_memory=True, dtype=torch.float32)
                    dst = torch.empty(n, device='cuda', dtype=torch.float32)
                    traffic = size
                else:
                    src = torch.ones(n, device='cuda', dtype=torch.float32)
                    dst = torch.empty(n, pin_memory=True, dtype=torch.float32)
                    traffic = size
                timing = measure(lambda: dst.copy_(src, non_blocking=True))
                torch.cuda.synchronize()
                if not bool((dst == 1).all().item()):
                    raise RuntimeError('Copy correctness check failed')
                return {**timing, 'buffer_bytes': size, 'traffic_bytes_per_iteration': traffic,
                        'GB_per_s': traffic / (timing['median_ms'] / 1000) / 1e9,
                        'correctness': 'all elements equal source'}

            for direction in ('device_copy', 'host_to_device', 'device_to_host'):
                run(direction, lambda direction=direction: copy_test(direction))

            def gemm(dtype):
                n = args.matrix_size
                available, _ = torch.cuda.mem_get_info()
                element_size = torch.empty((), dtype=dtype).element_size()
                if 3 * n * n * element_size > available * .5:
                    raise RuntimeError('Matrix buffers exceed 50% of free memory; reduce --matrix-size')
                a = torch.randn(n, n, device='cuda', dtype=dtype)
                b = torch.randn(n, n, device='cuda', dtype=dtype)
                c = torch.empty_like(a)
                timing = measure(lambda: torch.mm(a, b, out=c))
                if not bool(torch.isfinite(c).all().item()):
                    raise RuntimeError('Non-finite GEMM output')
                # Small independent CPU double-precision reference, outside timed region.
                reference = a[:8].cpu().double() @ b[:, :8].cpu().double()
                actual = c[:8, :8].cpu().double()
                relative_error = ((actual-reference).norm() / reference.norm().clamp_min(1e-30)).item()
                tolerance = {torch.float32: 1e-3, torch.float16: 1e-2, torch.bfloat16: 5e-2}[dtype]
                if relative_error > tolerance:
                    raise RuntimeError('GEMM reference error %.6g exceeds %.6g' % (relative_error, tolerance))
                return {**timing, 'matrix_size': n, 'dtype': str(dtype), 'tf32_allowed': False,
                        'TFLOP_per_s': 2 * n**3 / (timing['median_ms'] / 1000) / 1e12,
                        'reference_relative_error': relative_error}

            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                run('gemm_' + str(dtype).split('.')[-1], lambda dtype=dtype: gemm(dtype))
        report['status'] = 'complete' if not report['errors'] else 'partial_failure'
    except Exception as exc:
        report['status'] = 'failed'
        report['errors'].append({'error': '%s: %s' % (type(exc).__name__, exc)})
        print(report['errors'][-1]['error'], file=sys.stderr)
    finally:
        stop.set()
        if monitor.is_alive():
            monitor.join(timeout=35)
        save()
    print('Report: ' + str(out.resolve()), flush=True)
    return 0 if report['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
