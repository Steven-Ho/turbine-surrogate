"""Summarize all array tasks, preserve failures, compare equal-stage training timing."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ['POD', 'UNet', 'GCN', 'Graph_UNet', 'Graph_Transformer', 'Transolver_3', 'GeoTransolver']


def stage_rows(model, status, baseline):
    rows = []
    previous_epochs = 0
    for name, stage in status.get('stages', {}).items():
        target = int(name)
        completed = stage.get('completed_epochs')
        added = None if completed is None else completed - previous_epochs
        local = baseline['models'][model]
        if target == 20:
            local_seconds = local['twenty_epoch'].get('training_seconds')
            local_added = 20
        else:
            local_seconds = local.get('long_stages', {}).get(name, {}).get('training_seconds')
            local_completed = local.get('long_completed_epochs', 0)
            local_added = max(0, min(target, local_completed) - (20 if target == 200 else 200))
        seconds = stage.get('training_seconds')
        row = {'model': model, 'stage_target_epochs': target, 'state': status['state'],
               'completed_epochs': completed, 'additional_epochs': added,
               'model_backend': 'portable_torch_radius_cached' if model == 'GeoTransolver' else 'unchanged',
               'cluster_training_seconds': seconds,
               'cluster_pipeline_train_seconds': stage.get('pipeline_train_seconds'),
               'cluster_seconds_per_additional_epoch': seconds / added if seconds is not None and added else None,
               'local_training_seconds': local_seconds,
               'local_additional_epochs': local_added if local_seconds is not None else None,
               'local_seconds_per_additional_epoch': local_seconds / local_added if local_seconds is not None and local_added else None,
               'final_lr': stage.get('final_lr'), 'exit_code': stage.get('exit_code')}
        a, b = row['local_seconds_per_additional_epoch'], row['cluster_seconds_per_additional_epoch']
        row['local_over_cluster_throughput_ratio'] = a / b if a is not None and b else None
        row['identical_epoch_budget'] = added == local_added if local_seconds is not None and added is not None else False
        rows.append(row)
        if completed is not None:
            previous_epochs = completed
    return rows


def score_original_nodes(run, baseline):
    import numpy as np
    import torch
    from compare_models import read_vtu, metrics
    best = torch.load(run / 'train/best_model.pt', map_location='cpu', weights_only=False)
    evaluation = best['evaluate_result_test']
    entries = evaluation['samples_evaluated']
    assert len(entries) == 100
    true, pred, names = [], [], []
    for entry in entries:
        source = Path(entry['source_file']).resolve()
        xyz, y, topology = read_vtu(source, True)
        pxyz, py, ptopology = read_vtu(Path(entry['evaluated_result_path']), True)
        np.testing.assert_array_equal(xyz, pxyz)
        assert topology == ptopology
        true.append(y)
        pred.append(py)
        names.append(source.name)
    with (run / 'input/test.csv').open() as f:
        expected = {Path(row['file_path']).name for row in csv.DictReader(f)}
    assert len(set(names)) == 100 and set(names) == expected
    # Same threshold used in the local report, fitted on the same training data.
    threshold = baseline['mape_absolute_threshold_pa']
    result = metrics(np.stack(true), np.stack(pred), threshold)
    result['best_optimizer_steps'] = best['global_step']
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--skip-metrics', action='store_true', help='Only aggregate timing and status; no torch/VTK imports')
    args = parser.parse_args()
    parent = (ROOT / args.run_root).resolve()
    baseline = json.loads((ROOT / 'configs/cluster_long/local_baseline.json').read_text())
    parent.mkdir(parents=True, exist_ok=True)
    states, timings, scores = {}, [], []
    for model in MODELS:
        run = parent / model
        status_path = run / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'model': model, 'state': 'not_started'}
        states[model] = status
        timings.extend(stage_rows(model, status, baseline))
        if status['state'] == 'complete' and not args.skip_metrics:
            try:
                scores.append({'model': model, **score_original_nodes(run, baseline)})
            except Exception as exc:
                states[model] = dict(status, scoring_error=str(exc))
    report = {'all_complete': all(s['state'] == 'complete' and 'scoring_error' not in s for s in states.values()),
              'models': states, 'timings': timings, 'original_node_metrics': scores,
              'timing_notes': [
                  'training_seconds: trainer logged loop time, including validation/checkpoint/data loading; excludes preprocessing.',
                  'pipeline_train_seconds: subprocess time including preprocessing; cluster cold cache differs from local warm cache.',
                  'Throughput ratio = local seconds/epoch divided by cluster seconds/epoch; >1 means cluster faster.',
                  'Different stopping epochs/LR schedules or GPU software versions limit causal hardware comparisons.',
                  'GeoTransolver uses a chunked/cached Torch radius-query backend; local baseline uses PhysicsNeMo/Warp. This backend difference affects timing and possibly capped-neighbor ordering.',
                  'Only POD, Transolver_3, GeoTransolver have actual local long-run baselines; other four only have 20-epoch baselines.',
              ]}
    (parent / 'comparison.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    for name, rows in [('timing_comparison.csv', timings), ('metrics_original_nodes.csv', scores)]:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with (parent / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    lines = ['# 集群长训练比较', '', '| 模型 | 状态 | 说明 |', '|---|---|---|']
    for model, status in states.items():
        error = status.get('error', status.get('scoring_error', ''))
        lines.append(f'| {model} | {status["state"]} | {error.replace(chr(10), " ")} |')
    lines += ['', '耗时对比见 timing_comparison.csv；原始节点指标见 metrics_original_nodes.csv。',
              '吞吐比大于 1 表示集群每 epoch 更快。不同早停轮数的行只作吞吐参考。',
              '没有本机长训记录的模型不会虚构长训基线。', '']
    (parent / 'REPORT.md').write_text('\n'.join(lines))
    print(parent / 'REPORT.md')
    if not report['all_complete']:
        sys.exit(1)


if __name__ == '__main__':
    main()
