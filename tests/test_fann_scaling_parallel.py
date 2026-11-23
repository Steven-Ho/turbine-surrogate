import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from scripts.summarize_fann_geo_scaling_parallel import summarize

ROOT=Path(__file__).resolve().parents[1]


class ParallelScalingTests(unittest.TestCase):
    def test_independent_node_submissions(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home);stub=root/'sbatch';trace=root/'calls.jsonl'
            stub.write_text('#!'+sys.executable+'\nimport os,json,sys\nwith open(os.environ["SUBMIT_TRACE"],"a") as f:f.write(json.dumps({"args":sys.argv[1:],"count":os.environ["FANN_SCALING_COUNTS"],"output":os.environ["FANN_SCALING_OUTPUT"]})+"\\n")\nprint(100+int(os.environ["FANN_SCALING_COUNTS"]))\n')
            stub.chmod(0o755)
            env=dict(os.environ,PATH=str(root)+':'+os.environ['PATH'],SUBMIT_TRACE=str(trace),FANN_SCALING_PARALLEL_ROOT=str(root/'suite'))
            result=subprocess.run(['bash','scripts/submit_fann_geo_scaling_parallel.sh'],cwd=ROOT,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([call['count'] for call in calls],['1','2','4','8'])
            self.assertEqual(len(set(call['output'] for call in calls)),4)
            for call in calls:
                self.assertIn('--exclusive',call['args']);self.assertIn('--gres=gpu:8',call['args'])
                self.assertFalse(any(arg.startswith('--dependency') for arg in call['args']))
            self.assertEqual(len((root/'suite/jobs.tsv').read_text().splitlines()),4)

    def test_merge_and_reject_inconsistent_settings(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            self.assertEqual(summarize(root)['state'],'incomplete')
            for count in (1,2,4,8):
                path=root/'parts'/f'{count}gpu';path.mkdir(parents=True)
                (path/'report.json').write_text(json.dumps({'state':'complete','dataset_signature':'same','warmup_cases':16,
                    'measure_cases':64,'torch_version':'same','device':'cuda','gpu_name':'same','host':f'node{count}',
                    'results':[{'gpus':count,'measured_cases':64,'cases_per_second':count/10}]}))
            report=summarize(root)
            self.assertEqual(report['state'],'complete')
            self.assertEqual(report['results'][-1]['speedup'],8.)
            self.assertEqual(report['results'][-1]['efficiency'],1.)
            last=root/'parts/8gpu/report.json';data=json.loads(last.read_text());data['measure_cases']=32;last.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'measure_cases'):summarize(root)
