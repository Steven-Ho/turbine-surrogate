"""Evaluate a frozen faNN checkpoint without fitting normalization or POD."""
import argparse
from pathlib import Path
from tfai.benchmarks.fann_training import inspect_splits, evaluate_frozen


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--split',choices=['valid','test'],default='test');p.add_argument('--device',default='cuda')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-legacy-inputs',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Use a new output directory')
    evaluate_frozen(a.checkpoint,inspect_splits(a.splits,a.allow_legacy_inputs),a.split,a.device,a.output)

if __name__=='__main__':main()
