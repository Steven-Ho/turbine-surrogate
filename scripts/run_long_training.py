"""Continue Rotor37 models with validation-only scheduling and automatic final testing."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import torch

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["Transolver_3", "GeoTransolver", "POD"]
SOURCES = {name: ROOT / ("runs/comparison20/POD" if name == "POD" else f"runs/transolver_comparison20/{name}") for name in MODELS}
LOCK = threading.Lock()


def extend_training(history, first_epochs=200, extension_epochs=400):
    """Extend only a complete first stage that still improved meaningfully late in training."""
    losses = history["val_loss"]
    units = history.get("val_units", [(i + 1) * 800 for i in range(len(losses))])
    if not units or units[-1] < first_epochs * 800 or first_epochs >= extension_epochs:
        return False
    split = (first_epochs - 40) * 800
    before = [loss for step, loss in zip(units, losses) if step <= split]
    recent = [loss for step, loss in zip(units, losses) if step > split]
    return bool(before and recent and min(recent) < .98 * min(before))


def verify_and_relocate_manifests(run, source):
    """Migrate location-dependent fingerprints only after proving unchanged data."""
    marker = run / "verified_cache_relocation.json"
    if marker.exists(): return
    from tfai.dataset.full_sample_dataset import FullSampleDataset
    from tfai.training.trainer import Trainer
    from tfai.utils.torch_utils import atomic_torch_save
    processed = run / "data/processed"
    original = source / "data/processed"
    linked = list(original.glob("sample_*.pt"))
    assert linked and all(os.path.samefile(path, processed / path.name) for path in linked)
    for filename in ['train.csv', 'valid.csv', 'effective_train.csv', 'effective_valid.csv']:
        assert (original / filename).read_bytes() == (processed / filename).read_bytes()
    migration = {"immutable_sample_files_verified": len(linked), "manifests": {}}
    for filename in ["best_model.pt", "last_checkpoint.pt"]:
        path = run / "train" / filename
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        assert checkpoint['global_step'] <= 16000, "Cannot migrate an already continued run"
        config = checkpoint['config']
        for phase, key in [('train', 'train_manifest'), ('valid', 'validation_manifest')]:
            before = FullSampleDataset.from_csv(str(original / f'effective_{phase}.csv'), config, str(original), phase=phase)
            after = FullSampleDataset.from_csv(str(processed / f'effective_{phase}.csv'), config, str(processed), phase=phase)
            assert before.sample_ids == after.sample_ids
            old_fingerprint = Trainer._build_dataset_fingerprint(before)
            new_fingerprint = Trainer._build_dataset_fingerprint(after)
            assert checkpoint[key]['fingerprint'] == old_fingerprint
            checkpoint[key]['fingerprint'] = new_fingerprint
            migration['manifests'][f'{filename}:{phase}'] = {'before':old_fingerprint,'after':new_fingerprint}
        config['general']['experiment'] = str(run)
        config['general']['train_processed_dir'] = str(processed)
        config['model']['checkpoint_path'] = str(run / 'train/best_model.pt')
        config['training']['resume']['path'] = str(run / 'train/last_checkpoint.pt')
        atomic_torch_save(checkpoint, str(path))
    marker.write_text(json.dumps(migration, indent=2) + "\n")


def prepare_run(run, model):
    source = SOURCES[model]
    processed = run / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for path in (source / "data/processed").iterdir():
        target = processed / path.name
        if not path.is_file() or target.exists(): continue
        if path.name.startswith("sample_") and path.suffix == ".pt":
            os.link(path, target)
        else:
            shutil.copy2(path, target)
    train = run / "train"
    train.mkdir(exist_ok=True)
    for filename in ["best_model.pt", "last_checkpoint.pt"]:
        if not (train / filename).exists():
            shutil.copy2(source / "train" / filename, train / filename)
    evaluation = run / "evaluate"
    if not evaluation.exists():
        cache = Path("/tmp/turbine-long-evaluation") / run.parent.name / model
        cache.mkdir(parents=True, exist_ok=True)
        if evaluation.is_symlink(): evaluation.unlink()
        evaluation.symlink_to(cache, target_is_directory=True)
    provenance = run / "continuation_source.json"
    if not provenance.exists():
        provenance.write_text(json.dumps({"source_run": str(source.relative_to(ROOT)),
            "source_last_sha256": hashlib.sha256((source / "train/last_checkpoint.pt").read_bytes()).hexdigest(),
            "starting_optimizer_steps": 16000, "starting_epochs": 20,
            "optimizer_and_rng_restored": True}, indent=2) + "\n")
    verify_and_relocate_manifests(run, source)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--run-root", default="runs/transolver_long")
    parser.add_argument("--first-epochs", type=int, default=200)
    parser.add_argument("--max-epochs", type=int, default=400)
    parser.add_argument("--probe", action="store_true", help="Resume each model for two full-grid updates and validate fully; no test set")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.first_epochs <= 20 or args.max_epochs < args.first_epochs:
        parser.error("Require 20 < first-epochs <= max-epochs")
    parent = ROOT / args.run_root
    if args.probe and parent == ROOT / "runs/transolver_long":
        parser.error("Use a separate --run-root for probes")
    if args.dry_run:
        print(json.dumps({"models": args.models, "first_epochs": args.first_epochs,
              "max_epochs": args.max_epochs, "run_root": str(parent), "probe": args.probe}, indent=2)); return
    parent.mkdir(parents=True, exist_ok=True)
    lock_handle = (parent / "runner.lock").open("a")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A long-training runner already holds this run directory")
    status_path = parent / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    def update(model, **values):
        with LOCK:
            status.setdefault(model, {}).update(values)
            temporary = status_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(status, indent=2) + "\n")
            temporary.replace(status_path)
    def execute(model):
        if status.get(model, {}).get("state") == "complete":
            return True
        gpu = MODELS.index(model)
        run = parent / model
        prepare_run(run, model)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUTF8="1", MPLBACKEND="Agg")
        try:
            for epochs in ([21] if args.probe else sorted(set([args.first_epochs, args.max_epochs]))):
                last = torch.load(run / "train/last_checkpoint.pt", map_location="cpu", weights_only=False)
                if epochs > args.first_epochs and not args.probe and not extend_training(last["history"], args.first_epochs, args.max_epochs):
                    update(model, extension=False, extension_reason="Late validation improvement below 2%, or early stopping before first-stage cap")
                    break
                if last["global_step"] >= epochs * 800: continue
                command = [sys.executable, "scripts/train.py", "--config-name", f"train_rotor37_{model.lower()}_long",
                           f"general.experiment={run}", f"training.epochs={epochs}",
                           f"hydra.run.dir={run}/hydra_train_{epochs}"]
                if args.probe:
                    command += ["training.max_steps=16002", "training.validate_every=1", "training.lr_scheduler.patience=0", "training.lr_scheduler.threshold=1.0", "training.lr_scheduler.cooldown=0"]
                update(model, state="training", error=None, gpu=gpu, target_epochs=epochs, command=command, stage_start=time.time())
                with (run / f"train_{epochs}_console.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    update(model, pid=process.pid)
                    code = process.wait()
                if code: raise RuntimeError(f"Training exited with {code}")
                last = torch.load(run / "train/last_checkpoint.pt", map_location="cpu", weights_only=False)
                assert last["global_step"] > 16000 and last["lr_scheduler_state_dict"] is not None
                assert all(torch.isfinite(v).all() for v in last["model_state_dict"].values() if isinstance(v, torch.Tensor))
                update(model, completed_steps=last["global_step"], completed_epochs=last["global_step"] / 800,
                       best_validation_loss=last["best_score"], lr=last["optimizer_state_dict"]["param_groups"][0]["lr"])
                if args.probe:
                    assert last["global_step"] == 16002
                    assert last["optimizer_state_dict"]["param_groups"][0]["lr"] < 1e-4
                    update(model, state="probe_complete", end_time=time.time()); return True
                milestone = Path("/tmp/turbine-long-milestones") / parent.name / model / f"milestone_{epochs}"
                milestone.mkdir(parents=True, exist_ok=True)
                for filename in ["best_model.pt", "last_checkpoint.pt"]:
                    target = milestone / filename
                    if not target.exists(): shutil.copy2(run / "train" / filename, target)
                if last["global_step"] < epochs * 800:
                    update(model, early_stopped=True); break
            # Test only after validation-driven training and checkpoint selection are finished.
            command = [sys.executable, "scripts/evaluate.py", f"evaluate.experiment={run}",
                       "evaluate.evaluate_samples=runs/rotor37/data/raw/test.csv",
                       "evaluate.strict_writeback=true", "evaluate.plot_scatter=false",
                       f"hydra.run.dir={run}/hydra_evaluate"]
            update(model, state="evaluating", command=command)
            with (run / "evaluate_console.log").open("w") as log:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                update(model, pid=process.pid)
                code = process.wait()
            if code: raise RuntimeError(f"Evaluation exited with {code}")
            update(model, state="complete", end_time=time.time())
            return True
        except Exception as error:
            update(model, state="failed", error=str(error), end_time=time.time())
            raise
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(execute, args.models))
    if not args.probe and set(args.models) == set(MODELS):
        subprocess.run([sys.executable, "scripts/report_long_training.py", "--run-root", args.run_root], cwd=ROOT, check=True)
    print(json.dumps(status, indent=2))
    if not all(results): raise SystemExit(1)


if __name__ == "__main__": main()
