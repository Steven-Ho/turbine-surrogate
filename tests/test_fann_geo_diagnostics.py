import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from scripts.diagnose_fann_geo import sequence
from tests.test_fann_formal import ROOT,fixture
from tfai.benchmarks.fann_geo_diagnostics import GeoQueryTrace
from tfai.models._vendor.geotransolver2.ball_query import BQWarp


class GeoDiagnosticTests(unittest.TestCase):
    def test_swap_sequence_and_trace_restore(self):
        self.assertEqual(sequence(0,0),['A','A','B','B','A'])
        self.assertEqual(sequence(1,0),sequence(0,1))
        with tempfile.TemporaryDirectory() as home:
            with unittest.mock.patch.dict(os.environ,{'TURBINE_GEO_CACHE_DIR':home,'TURBINE_GEO_REFERENCE_MAX_POINTS':'0'}):
                module=BQWarp(radius=.5,neighbors_in_radius=2)
            xyz=torch.tensor([[0.,0,0],[1.,0,0]])
            original=module._indices
            with GeoQueryTrace(module) as trace:
                trace.reset(True);a=module._indices(xyz,xyz)
                b=module._indices(xyz,xyz)
                self.assertEqual(trace.events[0]['tree_builds'],1)
                self.assertEqual(trace.events[1]['device_cache_hits'],1)
                self.assertEqual(trace.events[0]['cache_access'][0]['source'],'miss')
                torch.testing.assert_close(a[0],b[0])
            self.assertEqual(module._indices,original)

    def test_suite_submits_sequential_jobs_and_same_allocation_offload_comparison(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);stub=root/'sbatch';trace=root/'submissions.txt'
            stub.write_text('#!/bin/bash\nprintf "%s|%s|%s|%s\\n" "$FANN_DIAG_OUTPUT" "$FANN_DIAG_GPUS_PER_NODE" "$FANN_DIAG_COMPARE_OFFLOAD" "$*" >> "$SUBMIT_TRACE"\nif [[ "$FANN_DIAG_GPUS_PER_NODE" == 8 ]]; then echo 100; else echo 101; fi\n')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],SUBMIT_TRACE=str(trace),FANN_DIAG_ROOT=str(root/'suite'))
            result=subprocess.run(['bash',str(ROOT/'scripts/submit_fann_geo_diagnostics.sh'),'--partition=test'],cwd=root,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            lines=trace.read_text().splitlines();self.assertEqual(len(lines),2)
            self.assertIn('|8|0|',lines[0]);self.assertIn('|1|1|',lines[1])
            self.assertIn('--dependency=afterok:100',lines[1])
            self.assertIn('--nodes=1',lines[1]);self.assertIn('--gres=gpu:1',lines[1])

    def test_fixed_weight_cpu_ddp_and_offload_variants(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);splits=fixture(root)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(ROOT),OMP_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1',TURBINE_GEO_NODE_CHUNK='7',TURBINE_GEO_REFERENCE_MAX_POINTS='0',
                TURBINE_GEO_CACHE_DIR=str(root/'neighbors'),TURBINE_FANN_STARTUP_CACHE=str(root/'startup'))
            for offload in ('on','off'):
                output=root/offload
                cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
                    'scripts/diagnose_fann_geo.py','--splits',str(splits),'--output',str(output),
                    '--device','cpu','--backend','gloo','--threads','1','--amp','bf16','--case-b','1','--offload',offload]
                result=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
                rows=[json.loads(line) for line in (output/'timings.jsonl').read_text().splitlines()]
                self.assertEqual(len(rows),10)
                rr=[row['ranks'][0] for row in rows]
                self.assertEqual([row['label'] for row in rr],sequence(0,0)+sequence(0,1))
                self.assertTrue(all(row['geo']['queries'] for row in rr))
                self.assertEqual(rr[0]['source'],rr[1]['source'])
                self.assertAlmostEqual(rr[0]['loss'],rr[1]['loss'],places=6)
                self.assertEqual(json.loads((output/'protocol.json').read_text())['optimizer_updates'],0)
                self.assertTrue(json.loads((output/'completion.json').read_text())['parameters_unchanged'])
                self.assertTrue((output/'summary.csv').is_file())
