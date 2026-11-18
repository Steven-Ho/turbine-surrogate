"""Run reproducible probes or equal-epoch Rotor37 comparison on three GPUs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["POD", "UNet", "GCN", "Graph_UNet", "Graph_Transformer", "Transolver_3", "GeoTransolver"]
GRAPH = {"GCN", "Graph_UNet", "Graph_Transformer", "Transolver_3", "GeoTransolver"}
LOCK = threading.Lock()

def seed_graph_cache(destination):
    destination.mkdir(parents=True, exist_ok=True)
    source = ROOT / "runs/rotor37/data/processed"
    for path in source.iterdir():
        if not path.is_file(): continue
        target = destination / path.name
        if target.exists(): continue
        if path.name.startswith("sample_") and path.suffix == ".pt":
            os.link(path, target)  # immutable normalized samples, independent metadata
        else:
            shutil.copy2(path, target)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["probe", "train", "evaluate"], required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--run-root", help="Alternate run directory relative to the project")
    parser.add_argument("--gpu", type=int, help="Use this GPU for selected model groups")
    parser.add_argument("--dry-run", action="store_true", help="Show run locations without starting jobs")
    args = parser.parse_args()
    parent = ROOT / "runs" / ("comparison_probes" if args.phase == "probe" else f"comparison{args.epochs}")
    if args.run_root: parent = ROOT / args.run_root
    parent.mkdir(parents=True, exist_ok=True)
    status_path = parent / "status.json"
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    override_file = parent / "run_overrides.json"
    overrides = json.loads(override_file.read_text()) if override_file.exists() else {}
    for model, entry in overrides.items():
        source_status = ROOT / entry["status"]
        if source_status.exists():
            states[model] = json.loads(source_status.read_text())[model]
    def update(model, **values):
        with LOCK:
            states.setdefault(model, {}).update(values)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(states, indent=2) + "\n")
            temporary.replace(status_path)
            if model in overrides:
                target_status = ROOT / overrides[model]["status"]
                target_status.parent.mkdir(parents=True, exist_ok=True)
                other = json.loads(target_status.read_text()) if target_status.exists() else {}
                other[model] = states[model]
                target_temp = target_status.with_suffix(".tmp")
                target_temp.write_text(json.dumps(other, indent=2) + "\n")
                target_temp.replace(target_status)
    def execute(model, gpu):
        run = ROOT / overrides[model]["run"] if model in overrides else parent / model
        if args.dry_run:
            print(f"{model}: phase={args.phase}, epochs={args.epochs}, gpu={gpu}, run={run}", flush=True)
            return True
        run.mkdir(parents=True, exist_ok=True)
        processed = run / "data/processed"
        if model == "UNet":
            if processed.is_symlink():
                processed.resolve().mkdir(parents=True, exist_ok=True)
            elif not processed.exists():
                cache = Path("/tmp/turbine-surrogate-cache") / parent.name / model / "processed"
                cache.mkdir(parents=True, exist_ok=True)
                processed.parent.mkdir(parents=True, exist_ok=True)
                processed.symlink_to(cache, target_is_directory=True)
        if model in {"Transolver_3", "GeoTransolver"}:
            # Prediction meshes and evaluation caches are regenerable; keep
            # training checkpoints in the project and large evaluation files on /tmp.
            evaluation = run / "evaluate"
            if not evaluation.exists() and not evaluation.is_symlink():
                cache = Path("/tmp/turbine-transolver-evaluation") / parent.name / model / "evaluate"
                cache.mkdir(parents=True, exist_ok=True)
                evaluation.symlink_to(cache, target_is_directory=True)
            elif evaluation.is_symlink():
                evaluation.resolve().mkdir(parents=True, exist_ok=True)
        if model in GRAPH: seed_graph_cache(run / "data/processed")
        train = [sys.executable, "scripts/train.py", "--config-name", f"train_rotor37_{model.lower()}",
                 f"general.experiment={run.relative_to(ROOT)}", f"training.epochs={args.epochs}",
                 f"hydra.run.dir={run}/hydra_train"]
        if args.phase == "probe":
            train += ["training.max_steps=2", "training.validate_every=1", "training.validation_max_batches=2"]
            if model not in GRAPH:
                train += ["general.train_samples=runs/rotor37_smoke/data/raw/samples.csv", "training.valid_ratio=0.25"]
        test_table = "runs/rotor37_smoke/data/raw/test.csv" if args.phase == "probe" else "runs/rotor37/data/raw/test.csv"
        evaluate = [sys.executable, "scripts/evaluate.py", f"evaluate.experiment={run.relative_to(ROOT)}",
                    f"evaluate.evaluate_samples={test_table}", f"hydra.run.dir={run}/hydra_evaluate",
                    "evaluate.strict_writeback=true", "evaluate.plot_scatter=false"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", MPLBACKEND="Agg")
        if args.phase != "evaluate":
            update(model, state="training", gpu=gpu, train_command=train, evaluate_command=evaluate,
                   epochs=args.epochs if args.phase == "train" else None, start_time=time.time())
            start = time.perf_counter()
            with (run / "train_console.log").open("w") as log:
                process = subprocess.Popen(train, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                update(model, pid=process.pid)
                code = process.wait()
            train_seconds = time.perf_counter() - start
            if code:
                update(model, state="failed", stage="train", returncode=code, train_wall_seconds=train_seconds)
                return False
        else:
            if not (run / "train/best_model.pt").exists():
                raise FileNotFoundError(run / "train/best_model.pt")
            train_seconds = states.get(model, {}).get("train_wall_seconds", 0.)
            update(model, evaluate_command=evaluate, gpu=gpu)
        update(model, state="evaluating", train_wall_seconds=train_seconds)
        start = time.perf_counter()
        with (run / "evaluate_console.log").open("w") as log:
            process = subprocess.Popen(evaluate, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            update(model, pid=process.pid)
            code = process.wait()
        update(model, state="complete" if code == 0 else "failed", stage="evaluate", returncode=code,
               evaluate_wall_seconds=time.perf_counter() - start, end_time=time.time())
        return code == 0
    # Each group has its own GPU; independent files and cache metadata avoid write races.
    groups = [(0, ["POD", "GCN", "Transolver_3"]), (1, ["UNet", "Graph_Transformer", "GeoTransolver"]), (2, ["Graph_UNet"])]
    def group(gpu, models):
        return [execute(m, gpu) for m in models if m in args.models]
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(group, args.gpu if args.gpu is not None else gpu, models) for gpu, models in groups]
        results = [item for future in as_completed(futures) for item in future.result()]
    if not args.dry_run: print(json.dumps(states, indent=2))
    if not all(results): raise SystemExit(1)

if __name__ == "__main__": main()
