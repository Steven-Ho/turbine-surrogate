"""Submission-only long-run checks; no Torch or accelerator runtime required."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class DomainLongSubmissionTests(unittest.TestCase):
    def test_resume_disables_budget_and_resets_stale_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);folder=root/'run/GeoTransolver';folder.mkdir(parents=True)
            for name in ['last.pt','best.pt']:(folder/name).touch()
            (folder/'status.json').write_text(json.dumps(dict(model='GeoTransolver',world_size=64,domain_size=4,
                state='interrupted',completed_epochs=5,position=0)))
            (folder/'history.json').write_text(json.dumps([dict(epoch=i,lr=1e-4) for i in range(1,6)]))
            stub=root/'sbatch';stub.write_text('#!/usr/bin/env python3\nimport os,json,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps({"env":dict(os.environ),"args":sys.argv[1:]}))\nprint(12345)\n');stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],TRACE=str(root/'trace'),
                FANN_MAX_EPOCHS_PER_RUN='5',FANN_MAX_UPDATES='20',FANN_STABILITY_DOMAIN_SIZE='1',FANN_STABILITY_RESUME='0')
            result=subprocess.run(['bash','scripts/submit_fann_geo_domain_long.sh','--run-root',str(root/'run')],
                                  cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'trace').read_text());e=report['env'];a=report['args']
            self.assertEqual(e['FANN_MAX_EPOCHS_PER_RUN'],'0');self.assertNotIn('FANN_MAX_UPDATES',e)
            self.assertEqual(e['FANN_EPOCHS'],'400');self.assertEqual(e['FANN_RESUME'],'1')
            self.assertEqual(e['FANN_DOMAIN_SIZE'],'4');self.assertEqual(e['FANN_RUN_ROOT'],str(root/'run'))
            self.assertIn('--nodes=8',a);self.assertIn('--exclusive',a);self.assertIn('--time=1-00:00:00',a)
            self.assertEqual(e['FANN_LEARNING_RATE'],'0.0001')
            self.assertEqual(e['FANN_BETA1'],'0.9');self.assertEqual(e['FANN_BETA2'],'0.999')
            self.assertEqual(e['FANN_AMP'],'bf16');self.assertEqual(e['TURBINE_GEO_ACTIVATION_OFFLOAD'],'0')
            self.assertIn('remaining<=395',result.stdout)
            # Refuse metadata for a job which is still running before submission.
            state=json.loads((folder/'status.json').read_text());state['state']='training'
            (folder/'status.json').write_text(json.dumps(state));(root/'trace').unlink()
            failed=subprocess.run(['bash','scripts/submit_fann_geo_domain_long.sh','--run-root',str(root/'run')],
                                  cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0);self.assertFalse((root/'trace').exists())
