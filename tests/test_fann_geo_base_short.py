"""No accelerator dependencies: verify matching base-model submission controls."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class BaseShortSubmissionTests(unittest.TestCase):
    def test_both_topologies_keep_sixteen_case_replicas(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stub=root/'sbatch'
            stub.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],TRACE=str(root/'trace'),
                FANN_STABILITY_LONG_RUN='1',FANN_STABILITY_RESUME='1',FANN_STABILITY_SCALE='large',
                FANN_MAX_UPDATES='2',FANN_STABILITY_EPOCHS='400')
            outputs=[]
            for nodes,domain in [(8,4),(16,8)]:
                with self.subTest(nodes=nodes):
                    result=subprocess.run(['bash',f'scripts/submit_fann_geo_base_{nodes}nodes.sh'],cwd=ROOT,
                                          env=env,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    report=json.loads((root/'trace').read_text());e=report['env'];a=report['args']
                    self.assertIn('--exclusive',a);self.assertIn(f'--nodes={nodes}',a)
                    self.assertIn('--gres=gpu:8',a);self.assertIn('--time=04:00:00',a)
                    self.assertEqual(e['FANN_SCALE'],'base');self.assertEqual(e['FANN_DOMAIN_SIZE'],str(domain))
                    self.assertEqual(nodes*int(e['FANN_DDP_GPUS_PER_NODE'])//domain,16)
                    self.assertEqual(e['FANN_MAX_EPOCHS_PER_RUN'],'5');self.assertEqual(e['FANN_RESUME'],'0')
                    self.assertEqual(e['FANN_EPOCHS'],'400');self.assertNotIn('FANN_MAX_UPDATES',e)
                    self.assertEqual(e['FANN_LEARNING_RATE'],'0.0001');self.assertEqual(e['FANN_BETA1'],'0.9')
                    self.assertEqual(e['FANN_BETA2'],'0.999');self.assertEqual(e['FANN_AMP'],'bf16')
                    self.assertEqual(e['TURBINE_GEO_ACTIVATION_OFFLOAD'],'0')
                    outputs.append(e['FANN_RUN_ROOT'])
            self.assertNotEqual(*outputs)
