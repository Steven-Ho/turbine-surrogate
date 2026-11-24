"""Resume both five-epoch six-node controls to a total of twenty epochs."""
import argparse
import os
from pathlib import Path
import subprocess
import torch


def check(root, lr):
    folder = root / 'GeoTransolver'
    checkpoint = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    if checkpoint['model'] != 'GeoTransolver':
        raise ValueError(f'{root}: wrong model')
    if abs(checkpoint['config']['learning_rate'] - lr) > 1e-12:
        raise ValueError(f'{root}: learning rate differs from {lr}')
    if checkpoint['completed_epochs'] != 5 or checkpoint['position'] != 0:
        raise ValueError(f'{root}: expected a complete epoch-5 checkpoint')
    if checkpoint.get('distributed_state', {}).get('world_size') != 48:
        raise ValueError(f'{root}: expected world_size=48')
    if not (folder / 'best.pt').is_file():
        raise ValueError(f'{root}: missing best.pt')
    return root


def resolve(explicit, pattern, lr):
    if explicit:
        return check(explicit, lr)
    candidates = []
    for root in sorted(Path('runs').glob(pattern)):
        if not (root / 'GeoTransolver' / 'last.pt').is_file():
            continue
        try:
            candidates.append(check(root, lr))
        except (ValueError, KeyError) as error:
            print(f'Skipping {error}', flush=True)
    if len(candidates) != 1:
        raise ValueError(f'Found {len(candidates)} matching runs for lr={lr}; specify --lr1-root and --lr17-root explicitly')
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lr1-root', type=Path)
    parser.add_argument('--lr17-root', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    # Validate both checkpoints before submitting either job. No dataset scan.
    runs = [
        (resolve(args.lr1_root, 'fann_geo_stability_6nodes_*', 1e-4), '0.0001', 'lr1'),
        (resolve(args.lr17_root, 'fann_geo_stability_6nodes_lr17_*', 1.7e-4), '0.00017', 'lr17'),
    ]
    if runs[0][0].resolve() == runs[1][0].resolve():
        raise ValueError('Experiments must have separate output directories')
    for root, lr, label in runs:
        env = dict(os.environ, FANN_STABILITY_NODES='6', FANN_STABILITY_EPOCHS='15',
                   FANN_STABILITY_RESUME='1', FANN_STABILITY_LEARNING_RATE=lr,
                   FANN_STABILITY_RUN_ROOT=str(root))
        # Do not propagate a CPU-only mask into actual GPU jobs.
        env.pop('CUDA_VISIBLE_DEVICES', None)
        command = ['bash', 'scripts/submit_fann_geo_stability.sh',
                   f'--job-name=fann_geo_{label}_20',
                   f'--output=logs/fann_geo_{label}_20_%A_%a.out',
                   f'--error=logs/fann_geo_{label}_20_%A_%a.err']
        print(f'{label}: resume {root}, epochs 6-20, lr={lr}', flush=True)
        if not args.dry_run:
            subprocess.run(command, env=env, check=True)


if __name__ == '__main__':
    main()
