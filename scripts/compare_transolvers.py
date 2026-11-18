"""Audit full-node Transolver predictions and merge the established five-model benchmark."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from compare_models import ROOT, csv_paths, read_vtu, metrics, training_loop_seconds


def main():
    output = ROOT / 'reports/transolver_comparison20'
    output.mkdir(parents=True, exist_ok=True)
    baseline_path = ROOT / 'reports/comparison20/comparison.json'
    report = json.loads(baseline_path.read_text())
    manifest = json.loads((ROOT / 'reports/comparison20/run_manifest.json').read_text())
    for entry in manifest['models']:
        assert hashlib.sha256((ROOT / entry['best_checkpoint']).read_bytes()).hexdigest() == entry['best_checkpoint_sha256']
    results = report['models']
    status = json.loads((ROOT / 'runs/transolver_comparison20/status.json').read_text())
    test = sorted(csv_paths(ROOT / 'runs/rotor37/data/raw/test.csv'))
    assert len(test) == len(set(test)) == 100
    original = [read_vtu(path, with_topology=True) for path in test]
    truth = np.stack([item[1] for item in original])
    assert truth.shape == (100, 29773)
    threshold = report['mape_absolute_threshold_pa']
    curves, predictions, sample_rows, audit = {}, {}, [], {}
    from tfai.factories.model_factory import ModelFactory
    for model in ['Transolver_3', 'GeoTransolver']:
        assert status[model]['state'] == 'complete'
        run = ROOT / 'runs/transolver_comparison20' / model
        last = torch.load(run / 'train/last_checkpoint.pt', map_location='cpu', weights_only=False)
        best = torch.load(run / 'train/best_model.pt', map_location='cpu', weights_only=False)
        assert last['global_step'] == 16000
        assert len(last['history']['val_loss']) == 20
        assert np.isfinite(last['history']['val_loss']).all()
        for split, count in [('train', 800), ('valid', 100)]:
            actual = set(csv_paths(run / f'data/processed/{split}.csv'))
            expected = set(csv_paths(ROOT / f'runs/rotor37/data/processed/{split}.csv'))
            assert len(actual) == count and actual == expected
            assert not actual.intersection(test)
        for value in best['model_state_dict'].values():
            if isinstance(value, torch.Tensor): assert torch.isfinite(value).all()
        config = best['config']
        assert config['model']['type'] == model
        assert config['general']['seed'] == 42
        assert config['training']['graph_use_partition'] is False
        assert config['training']['learning_rate'] == .0001
        assert config['training']['batch_size'] == 1
        assert config['training']['validation_max_batches'] is None
        evaluation = best['evaluate_result_test']
        assert evaluation['writeback_report']['evaluated']['success_count'] == 100
        entries = {Path(row['source_file']).resolve(): row for row in evaluation['samples_evaluated']}
        assert set(entries) == set(test)
        values = []
        for path, (coords, target, topology) in zip(test, original):
            predcoords, pred, predtopology = read_vtu(entries[path]['evaluated_result_path'], with_topology=True)
            np.testing.assert_array_equal(coords, predcoords)
            assert predtopology == topology
            values.append(pred)
            sample_rows.append({'model': model, 'sample': path.stem, **metrics(target, pred, threshold)})
        prediction = np.stack(values)
        predictions[model] = prediction
        instance = ModelFactory.create_model(model, 'field', config)
        parameters = sum(p.numel() for p in instance.parameters())
        del instance
        result = {'model': model, **metrics(truth, prediction, threshold), 'epochs': 20,
                  'optimizer_steps': last['global_step'], 'best_epoch': int(best['global_step'] // 800),
                  'parameters': parameters, 'training_seconds': training_loop_seconds((run / 'train/train.log').read_text()),
                  'pipeline_train_seconds': status[model]['train_wall_seconds'],
                  'pipeline_test_seconds': status[model]['evaluate_wall_seconds'], 'gpu': status[model]['gpu'],
                  'run_directory': str(run.relative_to(ROOT))}
        result['rmse_reduction_vs_mean_percent'] = 100 * (1 - result['rmse_pa'] / report['training_mean_field_baseline']['rmse_pa'])
        results.append(result)
        curves[model] = last['history']
        audit[model] = {'best_checkpoint_sha256': hashlib.sha256((run / 'train/best_model.pt').read_bytes()).hexdigest(),
                        'all_source_coordinates_and_topology_preserved': True, 'test_cases': 100,
                        'test_nodes': int(prediction.size), 'config': config}
        np.savez_compressed(output / f'{model}_predictions.npz', sample_names=np.array([p.name for p in test]),
                            pressure_pa=prediction.astype(np.float32))
    results.sort(key=lambda r: r['rmse_pa'])
    report['models'] = results
    report['baseline_report_sha256'] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    report['new_model_audit'] = audit
    report['geometry_protocol'] = 'Reference geometry inputs, matching prior five models; each prediction written to its original source mesh.'
    (output / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    keys = list(dict.fromkeys(k for row in results for k in row))
    with (output / 'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys); writer.writeheader(); writer.writerows(results)
    with (output / 'per_sample_metrics_new_models.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(sample_rows[0])); writer.writeheader(); writer.writerows(sample_rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = [r['model'] for r in results]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, key, label in zip(axes, ['mape_percent', 'rmse_pa', 'r2'], ['MAPE (%)', 'RMSE (Pa)', 'R-squared']):
        ax.bar(names, [r[key] for r in results]); ax.set_ylabel(label); ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=.25)
    fig.savefig(output / 'metrics.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for ax, (name, history) in zip(axes, curves.items()):
        for key in ['train_loss', 'val_loss']:
            if key in history: ax.plot(np.arange(1, len(history[key])+1), history[key], label=key)
        ax.set_title(name); ax.set_xlabel('Epoch'); ax.set_yscale('log'); ax.legend(); ax.grid(alpha=.25)
    fig.savefig(output / 'learning_curves.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout='constrained')
    coords = original[0][0]
    fields = [('CFD Pressure', truth[0])] + [(name, pred[0]) for name, pred in predictions.items()]
    for ax, (name, field) in zip(axes, fields):
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=field, s=1, vmin=truth[0].min(), vmax=truth[0].max(), cmap='turbo')
        ax.set_title(name); ax.set_aspect('equal'); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    fig.colorbar(scatter, ax=list(axes), label='Pressure (Pa)')
    fig.savefig(output / 'pressure_fields.png', dpi=160); plt.close(fig)
    lines = ['# Rotor37 七模型比较', '',
             '新增模型均使用官方实现：Transolver-3 固定源码提交，GeoTransolver 使用 PhysicsNemo 2.0.0。出处与依赖见 ../../docs/transolver_rotor37.md。', '',
             '相同 800/100/100 划分，seed=42、batch=1、学习率 1e-4；20 epoch、16000 次更新。按验证损失选择检查点；独立测试 100 个工况，共 2,977,300 个原始节点。旧五模型数值沿用已审核报告，新两模型重新读取原始 VTU 计算。', '',
             '| 模型 | MAPE (%) | RMSE (Pa) | R² | 最佳 epoch | 参数量 | 训练秒数 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['model']} | {r['mape_percent']:.3f} | {r['rmse_pa']:.1f} | {r['r2']:.4f} | {r['best_epoch']} | {r['parameters']:,} | {r['training_seconds']:.1f} |")
    mean = report['training_mean_field_baseline']
    new_interpretation = []
    for name in ['Transolver_3', 'GeoTransolver']:
        row = next(r for r in results if r['model'] == name)
        direction = '低' if row['rmse_pa'] < mean['rmse_pa'] else '高'
        difference = abs(100 * (row['rmse_pa'] / mean['rmse_pa'] - 1))
        history = curves[name]['val_loss']
        new_interpretation.append(f"{name} 的测试 RMSE 比平均场{direction} {difference:.1f}%；验证损失从第 15 轮 {history[14]:.6g} 变为第 20 轮 {history[19]:.6g}。")
    lines += ['', f"训练集平均场基线：MAPE {mean['mape_percent']:.3f}%，RMSE {mean['rmse_pa']:.1f} Pa。", '',
              ' '.join(new_interpretation), '',
              '输入为参考网格坐标与 Omega、P 两个工况量，输出为全体网格节点的 Pressure；本数据实验未训练 u、v、w、T。GeoTransolver 将坐标提供给几何/空间分支，工况作为全局上下文。Transolver-3 使用官方全局物理切片与分块实现。', '',
              '本轮沿用旧五模型的参考几何协议。各源网格坐标虽有变化，未作为逐样本变化的几何输入；因此这是 Rotor37 工况代理基准，不能证明变叶型、变角度或变冷却孔泛化。所有测试预测均验证与原始源网格坐标、连接关系一致。', '',
              '同 epoch 不等于同计算量；small 配置参数量不同。这是单种子预算内比较，不能证明普遍架构优劣或已经收敛。MAPE 阈值统一由训练集压力尺度确定。训练耗时不含预处理和独立测试。', '',
              '![统一指标](metrics.png)', '', '![新增模型学习曲线](learning_curves.png)', '', '![固定首个测试样本压力场](pressure_fields.png)', '',
              '审计及数值：comparison.json、metrics.csv、per_sample_metrics_new_models.csv。新增模型的完整测试预测另存 *_predictions.npz，按 sample_names 对齐原始数据；检查点和训练日志位于 runs/transolver_comparison20。']
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(output / 'REPORT.md')


if __name__ == '__main__':
    main()
