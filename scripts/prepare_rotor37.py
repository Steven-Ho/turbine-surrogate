"""Create deterministic train/validation pool and independent holdout tables."""
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    source = ROOT / "exp_rotor37_GCN/data/raw/samples.csv"
    with source.open() as f:
        rows = list(csv.DictReader(f))
    paths = [(source.parent / row["file_path"]).resolve() for row in rows]
    if len(set(paths)) != len(rows) or not all(p.is_file() for p in paths):
        raise ValueError("Duplicate or missing sample files")
    random.Random(42).shuffle(rows)
    for name, pool, holdout in [("rotor37", rows[100:], rows[:100]),
                                ("rotor37_smoke", rows[100:108], rows[:2])]:
        directory = ROOT / "runs" / name / "data/raw"
        directory.mkdir(parents=True, exist_ok=True)
        for filename, subset in [("samples.csv", pool), ("test.csv", holdout)]:
            with (directory / filename).open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                for row in subset:
                    row = dict(row)
                    row["file_path"] = "../../../../data/rotor37/" + Path(row["file_path"]).name
                    writer.writerow(row)
        assert not {r["file_path"] for r in pool} & {r["file_path"] for r in holdout}
        print(f"{name}: train+validation={len(pool)}, independent test={len(holdout)}")

if __name__ == "__main__":
    main()
