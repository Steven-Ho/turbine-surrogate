import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class InteropAllReduceTests(unittest.TestCase):
    def test_cpu_allreduce_and_interop_snapshot(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);output=root/'run'
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),FANN_INTEROP_THREADS='1',
                     FANN_COMM_DIAGNOSTICS_DIR=str(output/'communication'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                 'scripts/benchmark_fann_allreduce.py','--output',str(output),'--device','cpu','--backend','gloo',
                 '--sizes','4','4096','--warmup','1','--iterations','3']
            result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['state'],'complete');self.assertEqual(report['world_size'],2)
            self.assertEqual(len(report['results']),2)
            for row in report['results']:
                self.assertGreater(row['mean_ms'],0);self.assertEqual(len(row['rank_timings']),2)
            for path in (output/'communication').glob('runtime_*.json'):
                self.assertEqual(json.loads(path.read_text())['interop_threads'],1)
            self.assertEqual(len(list((output/'communication').glob('runtime_*.json'))),4)

    def test_checks_submitted_without_dependency_and_exclusively(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);stub=root/'sbatch';trace=root/'calls'
            stub.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$SUBMIT_TRACE"\necho 12345\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],SUBMIT_TRACE=str(trace),FANN_CHECKS_ROOT=str(root/'suite'))
            result=subprocess.run(['bash','scripts/submit_fann_interop_allreduce.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=trace.read_text().splitlines();self.assertEqual(len(calls),2)
            self.assertTrue(all('--exclusive' in line and '--dependency' not in line for line in calls))
