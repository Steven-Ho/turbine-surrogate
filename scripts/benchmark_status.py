"""Show saved job state and recent training progress without launching work."""
import argparse
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
args = parser.parse_args()
base = ROOT / "runs" / f"comparison{args.epochs}"
status = json.loads((base / "status.json").read_text())
overrides_path = base / "run_overrides.json"
overrides = json.loads(overrides_path.read_text()) if overrides_path.exists() else {}
for model, entry in overrides.items():
    path = ROOT / entry["status"]
    if path.exists(): status[model] = json.loads(path.read_text())[model]
    else: status[model] = {"state": "pending"}
for model in ["POD", "UNet", "GCN", "Graph_UNet", "Graph_Transformer"]:
    state = status.get(model, {})
    run = ROOT / overrides[model]["run"] if model in overrides else base / model
    log = run / "train_console.log"
    detail = "not yet launched"
    if log.exists():
        text = log.read_text()
        cycles = re.findall(r"验证周期完成:.*global_step=(\d+).*train_loss=([\d.]+) val_loss=([\d.]+)", text)
        steps = re.findall(r"训练步骤完成: global_step=(\d+)/(\d+)", text)
        detail = f"steps {steps[-1][0]}/{steps[-1][1]}" if steps else text.splitlines()[-1]
        if cycles: detail += f"; last validation loss {cycles[-1][2]} at step {cycles[-1][0]}"
    print(f"{model}: {state.get('state', 'pending')}; {detail}")
