"""Submission-only check; no GPU or Torch imports on the login node."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class TwoNodeCR15Tests(unittest.TestCase):
    def test_one_complete_epoch_with_sixteen_case_batch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);stub=root/'sbatch'
            stub.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=d+':'+os.environ['PATH'],TRACE=str(root/'trace'),
                CR15_RESUME='1',CR15_MAX_UPDATES='2',CR15_LARGEST_FIRST='1',CR15_GPUS_PER_NODE='1',
                CR15_RUN_ROOT='old_single_gpu',FANN_SPLITS='unchanged')
            result=subprocess.run(['bash','scripts/submit_cr15_2nodes_one_epoch.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'trace').read_text());e=report['env'];a=report['args']
            for value in ['--nodes=2','--gres=gpu:8','--cpus-per-task=32','--mem=64G','--exclusive']:
                self.assertIn(value,a)
            self.assertEqual(e['CR15_GPUS_PER_NODE'],'8');self.assertEqual(e['CR15_BATCH_CASES'],'16')
            self.assertEqual(e['CR15_EPOCH_BUDGET'],'1');self.assertEqual(e['CR15_EPOCHS'],'400')
            self.assertEqual(e['CR15_RESUME'],'0');self.assertEqual(e['CR15_LARGEST_FIRST'],'0')
            self.assertNotIn('CR15_MAX_UPDATES',e);self.assertEqual(e['CR15_SCALE'],'base')
            self.assertEqual(e['CR15_AMP'],'bf16');self.assertEqual(e['CR15_LEARNING_RATE'],'0.0001')
            self.assertNotEqual(e['CR15_RUN_ROOT'],'old_single_gpu');self.assertEqual(e['FANN_SPLITS'],'unchanged')
