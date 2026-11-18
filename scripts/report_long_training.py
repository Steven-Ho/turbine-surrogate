"""Audit final held-out predictions and compare extended runs with POD."""
import argparse
import hashlib
import json
import csv
from pathlib import Path
import numpy as np
import torch
from compare_models import ROOT, csv_paths, read_vtu, metrics


def paired_rmse_ratio_ci(prediction, reference, target, draws=2000):
    candidate_sse = np.sum((prediction - target) ** 2, axis=1)
    reference_sse = np.sum((reference - target) ** 2, axis=1)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(target), size=(draws, len(target)))
    ratios = np.sqrt(candidate_sse[indices].sum(axis=1) / reference_sse[indices].sum(axis=1))
    return [float(x) for x in np.quantile(ratios, [.025, .975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="runs/transolver_long")
    args = parser.parse_args()
    base = ROOT / args.run_root
    output = ROOT / "reports" / base.name
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((base / "status.json").read_text())
    previous = json.loads((ROOT / "reports/transolver_comparison20/comparison.json").read_text())
    threshold = previous['mape_absolute_threshold_pa']
    test = sorted(csv_paths(ROOT / 'runs/rotor37/data/raw/test.csv'))
    assert len(test) == len(set(test)) == 100
    source = [read_vtu(path, True) for path in test]
    target = np.stack([item[1] for item in source])
    assert target.shape == (100, 29773)
    predictions, results, audit, curves, per_sample = {}, [], {}, {}, []
    for model in ['POD', 'GeoTransolver', 'Transolver_3']:
        assert status[model]['state'] == 'complete'
        run = base / model
        last = torch.load(run / 'train/last_checkpoint.pt', map_location='cpu', weights_only=False)
        best_path = run / 'train/best_model.pt'
        best = torch.load(best_path, map_location='cpu', weights_only=False)
        assert last['global_step'] > 16000
        assert last['lr_scheduler_state_dict'] is not None
        for split, count in [('train',800), ('valid',100)]:
            paths = set(csv_paths(run / f'data/processed/{split}.csv'))
            assert len(paths) == count and not paths.intersection(test)
            assert paths == set(csv_paths(ROOT / f'runs/rotor37/data/processed/{split}.csv'))
        assert np.isfinite(last['history']['val_loss']).all()
        assert all(torch.isfinite(v).all() for v in best['model_state_dict'].values() if isinstance(v, torch.Tensor))
        evaluation = best['evaluate_result_test']
        entries = {Path(row['source_file']).resolve(): row for row in evaluation['samples_evaluated']}
        assert set(entries) == set(test)
        assert evaluation['writeback_report']['evaluated']['success_count'] == 100
        values = []
        for path, (coords, truth, topology) in zip(test, source):
            pc, pv, pt = read_vtu(entries[path]['evaluated_result_path'], True)
            np.testing.assert_array_equal(coords, pc); assert topology == pt
            values.append(pv)
            per_sample.append({'model': model, 'sample': path.stem, **metrics(truth, pv, threshold)})
        prediction = np.stack(values)
        predictions[model] = prediction
        np.savez_compressed(output / f'{model}_predictions.npz', sample_names=np.array([p.name for p in test]), pressure_pa=prediction.astype(np.float32))
        results.append({'model':model, **metrics(target,prediction,threshold),
                        'completed_epochs':last['global_step']/800, 'best_epoch':best['global_step']/800,
                        'optimizer_steps':last['global_step'], 'run_directory':str(run.relative_to(ROOT)),
                        'best_validation_loss':best['val_loss'], 'final_lr':last['optimizer_state_dict']['param_groups'][0]['lr']})
        audit[model] = {'best_checkpoint_sha256':hashlib.sha256(best_path.read_bytes()).hexdigest(),
                        'all_test_source_coordinates_and_topology_verified':True,
                        'config':last['config'], 'scheduler_state':last['lr_scheduler_state_dict']}
        curves[model] = last['history']
    original_pod = next(r for r in previous['models'] if r['model']=='POD')
    strong_pod = next(r for r in results if r['model']=='POD')
    for row in results:
        row['rmse_reduction_vs_pod20_percent'] = 100*(1-row['rmse_pa']/original_pod['rmse_pa'])
        row['rmse_reduction_vs_pod_long_percent'] = 100*(1-row['rmse_pa']/strong_pod['rmse_pa'])
        if row['model'] != 'POD':
            interval = paired_rmse_ratio_ci(predictions[row['model']],predictions['POD'],target)
            row['paired_case_bootstrap_rmse_ratio_95ci_vs_pod_long'] = interval
            row['target_achieved'] = bool(row['rmse_pa'] <= .8*original_pod['rmse_pa'] and interval[1] <= .8)
    report = {'protocol':{'train':800,'valid':100,'test':100,'seed':42,'selection':'validation only',
                          'reference_geometry_inputs':True,'test_nodes':int(target.size),
                          'bootstrap_unit':'held-out case, paired; fixed seed42; 2000 draws',
                          'target':'RMSE <= 80% of POD20 AND upper paired-case 95% RMSE-ratio CI <= 0.8 vs extended POD'},
              'original_pod':original_pod, 'models':results, 'audit':audit,
              'historical_test_set_already_used':True,
              'limitation':'The reused historical test set is not a new blind external set; no pipeline scheduling or checkpoint selection uses it.'}
    (output/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    keys=list(dict.fromkeys(key for row in results for key in row))
    with (output/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(results)
    with (output/'per_sample_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(per_sample[0]));writer.writeheader();writer.writerows(per_sample)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
    for ax,(name,history) in zip(axes,curves.items()):
        units=np.asarray(history['val_units'])/800
        ax.plot(units,history['train_loss'],label='train');ax.plot(units,history['val_loss'],label='validation')
        ax.axvline(20,color='gray',ls='--');ax.set_yscale('log');ax.set_title(name);ax.set_xlabel('Total epochs');ax.legend();ax.grid(alpha=.3)
    fig.savefig(output/'learning_curves.png',dpi=160);plt.close(fig)
    lines=['# Rotor37 长训练结果','',
           '沿用 800/100/100 划分，从原 20 轮检查点恢复权重、Adam 状态和随机数状态。仅用验证损失调度学习率、早停及选择检查点；最终一次测试 100 个历史保留工况。','',
           '| 模型 | 已完成轮数 | 最佳轮数 | MAPE (%) | RMSE (Pa) | R² |',
           '|---|---:|---:|---:|---:|---:|']
    lines.append(f"| POD（原 20 轮） | 20 | 20 | {original_pod['mape_percent']:.3f} | {original_pod['rmse_pa']:.1f} | {original_pod['r2']:.4f} |")
    for row in sorted(results,key=lambda r:r['rmse_pa']):
        lines.append(f"| {row['model']}（长训练） | {row['completed_epochs']:.0f} | {row['best_epoch']:.0f} | {row['mape_percent']:.3f} | {row['rmse_pa']:.1f} | {row['r2']:.4f} |")
    for row in results:
        if row['model']=='POD':continue
        interval=row['paired_case_bootstrap_rmse_ratio_95ci_vs_pod_long']
        lines += ['',f"{row['model']}：相对长训练 POD 的 RMSE 变化为降低 {row['rmse_reduction_vs_pod_long_percent']:.1f}%（负值表示更差）；按工况配对 bootstrap 的 RMSE 比值 95% 区间 [{interval[0]:.3f}, {interval[1]:.3f}]。预定显著提升目标{'达到' if row['target_achieved'] else '尚未达到'}。"]
    lines += ['', '![训练曲线](learning_curves.png)', '',
              '目标预先定义为：RMSE 比原 POD 至少低 20%，且相对长训练 POD 的工况配对 95% 置信区间上界不超过 0.8。该区间反映测试工况抽样不确定性，不代表多随机种子训练不确定性。', '',
              '本轮是长训练与原预算的比较，不是相同训练预算的架构排名。仍为参考几何到节点压力预测，不能据此推断新叶型/新冷却孔泛化。历史测试集此前已查看，本轮不用于调参，但仍需新增盲测数据验证最终结论。', '',
              '完整指标、检查点哈希及配置见 comparison.json；逐工况指标见 per_sample_metrics.csv；完整节点预测见 *_predictions.npz。']
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(output/'REPORT.md')


if __name__=='__main__':main()
