"""Score every model on the same original VTU nodes and export comparison artifacts."""
import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
import vtk
from vtk.util.numpy_support import vtk_to_numpy

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["POD", "UNet", "GCN", "Graph_UNet", "Graph_Transformer"]


def read_vtu(path, with_topology=False):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    pressure = grid.GetPointData().GetArray("Pressure")
    if pressure is None: raise ValueError(f"Pressure missing: {path}")
    result = (vtk_to_numpy(grid.GetPoints().GetData()).copy(), vtk_to_numpy(pressure).astype(np.float64))
    if with_topology:
        import hashlib
        digest = hashlib.sha256()
        for array in [grid.GetCells().GetConnectivityArray(), grid.GetCells().GetOffsetsArray(), grid.GetCellTypesArray()]:
            digest.update(vtk_to_numpy(array).tobytes())
        return (*result, digest.hexdigest())
    return result


def metrics(target, prediction, threshold):
    target, prediction = np.asarray(target), np.asarray(prediction)
    if target.shape != prediction.shape: raise ValueError("Prediction/target shape mismatch")
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError("Non-finite original-node predictions or targets")
    error = prediction - target
    valid = np.abs(target) >= threshold
    sse = np.sum(error ** 2)
    rmse = float(np.sqrt(np.mean(error ** 2)))
    return {"mae_pa": float(np.mean(np.abs(error))), "rmse_pa": rmse,
            "nrmse_range": rmse / float(np.ptp(target)),
            "relative_l2_percent": float(100 * np.sqrt(sse / np.sum(target ** 2))),
            "mape_percent": float(100 * np.mean(np.abs(error[valid] / target[valid]))),
            "mape_valid_count": int(valid.sum()),
            "r2": float(1 - sse / np.sum((target - target.mean()) ** 2)),
            "node_count": int(target.size)}


def training_loop_seconds(log):
    lines = re.findall(r"模型训练耗时: ([^\n]+)", log)
    if not lines:
        raise ValueError("Training duration was not logged")
    scales = {"小时": 3600., "时": 3600., "分": 60., "秒": 1.}
    parts = re.findall(r"([0-9.]+)\s*(小时|时|分|秒)", lines[-1])
    if not parts:
        raise ValueError(f"Unknown training duration format: {lines[-1]}")
    return sum(float(value) * scales[unit] for value, unit in parts)


def csv_paths(path):
    with path.open() as f:
        return [(path.parent / row["file_path"]).resolve() for row in csv.DictReader(f)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    base = ROOT / "runs" / f"comparison{args.epochs}"
    output = ROOT / "reports" / f"comparison{args.epochs}"
    output.mkdir(parents=True, exist_ok=True)
    status = json.loads((base / "status.json").read_text())
    assert all(status.get(m, {}).get("state") == "complete" for m in MODELS), "Training/test jobs unfinished"
    overrides_path = base / "run_overrides.json"
    overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
    for model, entry in overrides.items():
        status[model] = json.loads((ROOT / entry["status"]).read_text())[model]
    assert all(status.get(m, {}).get("state") == "complete" for m in MODELS), "Corrected jobs unfinished"
    test = sorted(csv_paths(ROOT / "runs/rotor37/data/raw/test.csv"))
    assert len(test) == len(set(test)) == 100
    coordinates, targets, topologies = zip(*(read_vtu(path, with_topology=True) for path in test))
    assert len(set(topologies)) == 1, "Reference-node comparison requires identical indexed topology"
    geometry_variation = max(float(np.max(np.abs(value - coordinates[0]))) for value in coordinates)
    target = np.stack(targets)
    assert target.shape == (100, 29773)
    norm = torch.load(ROOT / "runs/rotor37/data/processed/norm_params.pt", map_location="cpu", weights_only=False)
    threshold = 0.001 * norm["mape_target_scale"]["by_variable"]["Pressure"]
    results, sample_results, predictions, curves = [], [], {}, {}
    mean_field_baseline = None
    from tfai.factories.model_factory import ModelFactory
    for model in MODELS:
        run = ROOT / overrides[model]["run"] if model in overrides else base / model
        last = torch.load(run / "train/last_checkpoint.pt", map_location="cpu", weights_only=False)
        best = torch.load(run / "train/best_model.pt", map_location="cpu", weights_only=False)
        train = set(csv_paths(run / "data/processed/train.csv"))
        valid = set(csv_paths(run / "data/processed/valid.csv"))
        assert len(train) == 800 and len(valid) == 100
        assert not (train & valid or train & set(test) or valid & set(test))
        assert train | valid | set(test) == set((ROOT / "data/rotor37").glob("*.vtu"))
        expected_train = set(csv_paths(ROOT / "runs/rotor37/data/processed/train.csv"))
        assert train == expected_train, "Models used different training splits"
        assert last["global_step"] == 800 * args.epochs
        assert len(last["history"]["val_loss"]) == args.epochs
        assert last["config"]["training"]["validation_max_batches"] is None
        assert np.isfinite(last["history"]["val_loss"]).all()
        assert best["config"]["model"]["type"] == model
        assert best["config"]["training"]["learning_rate"] == 0.0001
        assert best["config"]["training"]["batch_size"] == 1
        evaluation = best["evaluate_result_test"]
        entries = {Path(row["source_file"]).resolve(): row for row in evaluation["samples_evaluated"]}
        assert set(entries) == set(test)
        assert evaluation["writeback_report"]["evaluated"]["success_count"] == 100
        model_predictions = []
        for index, path in enumerate(test):
            coords, values, topology = read_vtu(entries[path]["evaluated_result_path"], with_topology=True)
            assert topology == topologies[index], "Prediction topology changed"
            np.testing.assert_array_equal(coords, coordinates[index])
            model_predictions.append(values)
            sample_results.append({"model": model, "sample": path.stem,
                                   **metrics(targets[index], values, threshold)})
        prediction = np.stack(model_predictions)
        predictions[model] = prediction
        model_instance = ModelFactory.create_model(model, "field", best["config"])
        parameter_count = sum(parameter.numel() for parameter in model_instance.parameters())
        del model_instance
        log = (run / "train/train.log").read_text()
        elapsed_seconds = training_loop_seconds(log)
        results.append({"model": model, **metrics(target, prediction, threshold),
                        "epochs": args.epochs, "optimizer_steps": last["global_step"],
                        "best_epoch": int(best["global_step"] // 800),
                        "parameters": parameter_count,
                        "training_seconds": elapsed_seconds,
                        "pipeline_train_seconds": status[model]["train_wall_seconds"],
                        "pipeline_test_seconds": status[model]["evaluate_wall_seconds"],
                        "gpu": status[model]["gpu"], "run_directory": str(run.relative_to(ROOT))})
        curves[model] = last["history"]
        if model == "UNet":
            from scipy.spatial import cKDTree
            state = best["processor_state"]
            centers = np.asarray(state["cached_voxel_centers"])
            mask = np.asarray(state["cached_mask"]).reshape(-1) == 1
            assert int(mask.sum()) > 1000, "Surface sampling support degenerated"
            tree = cKDTree(centers[mask])
            reference_coords, _ = read_vtu(run / "train/reference.vtu")
            reference_distance, _ = tree.query(reference_coords)
            spacing = [np.min(np.diff(np.unique(centers[:, axis]))) for axis in range(3)]
            radius = .5 * np.linalg.norm(spacing)
            assert reference_distance.max() <= radius + 1e-9
            test_distance = np.concatenate([tree.query(coords)[0] for coords in coordinates])
            results[-1]["surface_valid_voxels"] = int(mask.sum())
            results[-1]["surface_reference_max_node_distance_m"] = float(reference_distance.max())
            results[-1]["surface_test_max_node_distance_m"] = float(test_distance.max())
            results[-1]["surface_test_nodes_outside_reference_band_percent"] = float(100 * np.mean(test_distance > radius))
        if model == "POD":
            mean_field = np.asarray(best["modes"]["mean_fields"]["Pressure"])
            mean_field_baseline = metrics(target, np.broadcast_to(mean_field, target.shape), threshold)
            ids = json.loads((run / "data/processed/pod_fit_samples.json").read_text())["sample_ids"]
            assert set(ids) == {int(path.stem.split("_")[-1]) for path in train}
        print(model, results[-1], flush=True)
        del last, best
    for row in results:
        row["rmse_reduction_vs_mean_percent"] = 100 * (1 - row["rmse_pa"] / mean_field_baseline["rmse_pa"])
    results.sort(key=lambda row: row["rmse_pa"])
    for filename, rows in [("metrics.csv", results), ("per_sample_metrics.csv", sample_results)]:
        with (output / filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(key for row in rows for key in row))); writer.writeheader(); writer.writerows(rows)
    report = {"data": "Rotor37", "split": {"train": 800, "valid": 100, "test": 100},
              "epochs": args.epochs, "seed": 42, "learning_rate": 0.0001, "batch_size": 1,
              "selection": "minimum validation loss within each model's own training representation",
              "scoring": "raw original Pressure on every test VTU node; includes POD truncation and UNet interpolation error",
              "mape_absolute_threshold_pa": threshold, "test_nodes": int(target.size),
              "models": results, "training_curves": curves, "training_mean_field_baseline": mean_field_baseline,
              "geometry_protocol": "shared indexed topology with reference-geometry training, original per-case geometry on writeback; not a geometry-generalization study",
              "test_max_coordinate_variation_m": geometry_variation,
              "test_indexed_topology_sha256": topologies[0],
              "single_seed": True, "equal_epoch_budget_not_equal_compute": True}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    make_figures(output, results, sample_results, curves, coordinates[0], targets[0],
                 {m: v[0] for m, v in predictions.items()}, test[0].stem)
    lines = ["# Rotor37 五模型预测效果对比", "",
             f"所有模型从头训练 {args.epochs} 轮（每轮 800 个工况、batch=1、共 {800 * args.epochs} optimizer steps），seed=42，学习率 1e-4，scale=small。",
             "使用相同的 800/100/100 划分，仅由验证集选择各自最佳检查点。所有模型都在独立测试集的原始 2,977,300 个节点上重新计算误差。",
             "POD 模态只拟合 800 个训练样本；UNet 最长轴网格分辨率为 50（实际网格形状见检查点），曲面数据采用半体素对角线宽度的表面带掩码，图模型保留完整网格。", "",
             "| 模型 | MAPE (%) ↓ | RMSE (Pa) ↓ | MAE (Pa) ↓ | R² ↑ | 最佳轮数 | 参数量 | 训练耗时 (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in results:
        lines.append(f"| {row['model']} | {row['mape_percent']:.3f} | {row['rmse_pa']:.1f} | {row['mae_pa']:.1f} | {row['r2']:.4f} | {row['best_epoch']} | {row['parameters']:,} | {row['training_seconds']:.1f} |")
    leader = results[0]
    gain = leader["rmse_reduction_vs_mean_percent"]
    gain_text = f"比训练集平均场低 {gain:.1f}%" if gain >= 0 else f"比训练集平均场高 {-gain:.1f}%"
    lines += ["", f"本轮最低 RMSE 为 {leader['model']}，{gain_text}。", "", f"训练集平均压力场参考（不使用 Omega/P）：MAPE {mean_field_baseline['mape_percent']:.3f}%，RMSE {mean_field_baseline['rmse_pa']:.1f} Pa。这用于判断是否学到了超出固定平均场的工况变化。", "", "在该预算下，优于训练集平均场 RMSE 的模型：" + ", ".join(row["model"] for row in results if row["rmse_pa"] < mean_field_baseline["rmse_pa"]) + "。", "", "![统一节点指标](metrics.png)", "", "![单样本压力场](pressure_fields.png)", "", "![绝对误差场](error_fields.png)", "", "![训练与验证曲线](learning_curves.png)", "",
              "## 解读与限制", "",
              "按 RMSE 排序；训练损失空间不同，不能直接用各模型训练 loss 排名。表中耗时为模型训练循环耗时，不含预处理和独立测试；三张同型号 GPU 并行运行，系统负载会影响耗时。完整流水线耗时保存在 CSV/JSON。",
              f"数据同一节点拓扑，但测试样本坐标存在变化（相对第一个测试样本最大坐标差 {geometry_variation:.6g} m）。本轮采用参考几何训练设定，POD 按共同节点编号建模，图模型使用参考几何特征，UNet 在参考体素网格上学习并按各测试样本坐标回采样；所有输出均保留该样本真实几何。未将独立几何参数作为工况输入，不能据此声称具有新几何泛化能力。",
              "这是单个种子、相同 epoch 预算下的首轮实验，未做多种子统计或逐模型超参数搜索；不能据此认定某种架构普遍更优。每个模型的 small 参数量不同，相同样本遍历次数不等于相同计算成本。",
              "UNet 的网格化与插值误差、POD 的降维重建误差均计入统一原始节点指标。MAPE 阈值统一采用训练集绝对压力 p95 的 0.001 倍，避免不同处理器阈值造成偏差。",
              f"压力场图选择固定测试样本 {test[0].stem}，仅为展示，不按模型效果挑选；散点图为清晰展示做固定步长抽稀，指标使用全部节点。",
              "", "原始结果：metrics.csv、per_sample_metrics.csv、comparison.json；模型及完整日志路径见 comparison.json 中 run_directory；原曲面处理有误的 UNet 历史运行已排除，最终使用 runs/comparison20_surface/UNet。"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(output / "REPORT.md")


def make_figures(output, results, samples, curves, coords, truth, predictions, sample_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130})
    names = [row["model"] for row in results]
    colors = plt.get_cmap("tab10")(np.arange(5))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for ax, key, label in zip(axes, ["mape_percent", "rmse_pa", "r2"], ["MAPE (%)", "RMSE (Pa)", "R-squared"]):
        ax.bar(names, [row[key] for row in results], color=colors)
        ax.set_ylabel(label); ax.tick_params(axis="x", rotation=35); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Rotor37: same 100 held-out cases, original mesh nodes")
    fig.savefig(output / "metrics.png"); fig.savefig(output / "metrics.svg"); plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), layout="constrained")
    for ax, model in zip(axes.flat, MODELS):
        history = curves[model]; epochs = np.arange(1, len(history["train_loss"]) + 1)
        ax.plot(epochs, history["train_loss"], label="train")
        ax.plot(epochs, history["val_loss"], label="validation")
        ax.set_title(model); ax.set_xlabel("Epoch"); ax.set_ylabel("Own representation loss"); ax.set_yscale("log"); ax.legend(); ax.grid(alpha=.2)
    axes.flat[-1].axis("off")
    fig.suptitle("Training diagnostics (loss scales differ across processors)")
    fig.savefig(output / "learning_curves.png"); plt.close(fig)
    # A fixed original-node sample, shared physical pressure color range.
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    values = {"Ground truth": truth, **predictions}
    vmin, vmax = float(truth.min()), float(truth.max())
    for index, (name, pressure) in enumerate(values.items(), start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        sl = slice(None, None, 3)
        image = ax.scatter(coords[sl, 0], coords[sl, 1], coords[sl, 2], c=pressure[sl],
                           s=2, cmap="viridis", vmin=vmin, vmax=vmax, rasterized=True)
        ax.set_title(name); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=25, azim=-55)
    fig.colorbar(image, ax=fig.axes, shrink=.6, label="Pressure (Pa)")
    fig.suptitle(f"Fixed held-out case: {sample_name}; shared pressure scale")
    fig.savefig(output / "pressure_fields.png"); plt.close(fig)
    fig = plt.figure(figsize=(14, 8), layout="constrained")
    maximum_error = max(float(np.max(np.abs(value - truth))) for value in predictions.values())
    for index, (name, pressure) in enumerate(predictions.items(), start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        sl = slice(None, None, 3)
        image = ax.scatter(coords[sl, 0], coords[sl, 1], coords[sl, 2],
                           c=np.abs(pressure[sl] - truth[sl]), s=2, cmap="magma",
                           vmin=0, vmax=maximum_error, rasterized=True)
        ax.set_title(name); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.view_init(elev=25, azim=-55)
    fig.colorbar(image, ax=fig.axes, shrink=.6, label="Absolute pressure error (Pa)")
    fig.suptitle(f"Fixed held-out case: {sample_name}; shared absolute error scale")
    fig.savefig(output / "error_fields.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
    ax.boxplot([[r["mape_percent"] for r in samples if r["model"] == name] for name in names], tick_labels=names)
    ax.set_ylabel("Per-case MAPE (%)"); ax.set_title("Variation across 100 test cases (one training seed)"); ax.grid(axis="y", alpha=.2)
    fig.savefig(output / "per_case_errors.png"); plt.close(fig)


if __name__ == "__main__": main()
