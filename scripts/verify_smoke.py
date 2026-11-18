"""Check the artifacts of real-data training and independent evaluation."""
import argparse
import csv
import json
import math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]

def paths(table):
    with table.open() as f:
        return {(table.parent / r["file_path"]).resolve() for r in csv.DictReader(f)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Verify the full 800/100/100 run")
    args = parser.parse_args()
    name = "rotor37" if args.full else "rotor37_smoke"
    RUN = ROOT / "runs" / name
    expected = (800, 100, 100) if args.full else (6, 2, 2)
    expected_steps = 800 if args.full else 2
    train = paths(RUN / "data/processed/train.csv")
    valid = paths(RUN / "data/processed/valid.csv")
    test = paths(RUN / "data/raw/test.csv")
    assert (len(train), len(valid), len(test)) == expected
    assert not (train & valid or train & test or valid & test)
    assert all(p.is_file() for p in train | valid | test)
    if args.full:
        assert train | valid | test == set((ROOT / "data/rotor37").glob("*.vtu"))
    saved_devices = set()
    def map_storage(storage, location):
        saved_devices.add(location)
        return storage
    best = torch.load(RUN / "train/best_model.pt", map_location=map_storage, weights_only=False)
    assert best["config"]["model"]["type"] == "GCN"
    assert best["config"]["task"]["variables"] == ["Pressure"]
    last = torch.load(RUN / "train/last_checkpoint.pt", map_location="cpu", weights_only=False)
    assert last["global_step"] == expected_steps
    assert last["steps_per_epoch"] == len(train)
    assert best["val_loss"] >= 0 and math.isfinite(best["val_loss"])
    for value in best["model_state_dict"].values():
        assert torch.isfinite(value).all(), "Non-finite model weights"
    result = best.get("evaluate_result_test")
    assert result and result.get("metrics"), "Independent evaluation missing"
    evaluated = result["samples_evaluated"]
    assert len(evaluated) == len(test)
    assert {Path(item["source_file"]).resolve() for item in evaluated} == test
    for item in evaluated:
        assert Path(item["evaluated_result_path"]).is_file()
        assert Path(item["target_result_path"]).is_file()
    writeback = result["writeback_report"]
    assert writeback["status"] == "success"
    assert writeback["evaluated"]["success_count"] == len(test)
    metrics = result["metrics"]
    def check_finite(value):
        if isinstance(value, dict):
            for v in value.values(): check_finite(v)
        elif isinstance(value, (float, int)):
            assert math.isfinite(value), value
    check_finite(metrics)
    report = {"model": "GCN", "variables": ["Pressure"],
              "split_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
              "disjoint": True, "test_metrics": metrics,
              "completed_optimizer_steps": last["global_step"],
              "best_validation_loss": best["val_loss"],
              "training_history": last["history"],
              "torch": torch.__version__, "checkpoint_keys": list(best),
              "saved_storage_devices": sorted(saved_devices),
              "evaluated_count": len(evaluated), "writeback_report": writeback}
    target = ROOT / "verification" / ("full_report.json" if args.full else "smoke_report.json")
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))

if __name__ == "__main__":
    main()
