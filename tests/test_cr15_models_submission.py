"""Login-node submission needs no Torch and clears inherited experiment settings."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class SubmissionTests(unittest.TestCase):
 def test_resource_profiles_dependencies_and_environment(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);prepared=root/'prepared';prepared.mkdir();(prepared/'summary.json').write_text(json.dumps(dict(state='complete',cache_ready=True)))
   log=root/'calls.jsonl';counter=root/'counter';tool=root/'sbatch'
   tool.write_text('#!'+sys.executable+'\nimport os,json,sys\nfrom pathlib import Path\np=Path('+repr(str(counter))+')\nn=int(p.read_text())+1 if p.exists() else 7000\np.write_text(str(n))\nwith open('+repr(str(log))+',"a") as f:f.write(json.dumps(dict(args=sys.argv[1:],env={k:v for k,v in os.environ.items() if k.startswith("CR15_") or k=="CUDA_VISIBLE_DEVICES"}))+"\\n")\nprint(n)\n');tool.chmod(0o755)
   env=dict(os.environ,PATH=tmp+':'+os.environ['PATH'],CUDA_VISIBLE_DEVICES='99',CR15_MAX_UPDATES='3',CR15_MODEL='GeoTransolver',CR15_RESUME='1',CR15_EPOCH_BUDGET='5')
   result=subprocess.run([sys.executable,'scripts/submit_cr15_models.py','--prepared',str(prepared),'--run-root',str(root/'runs')],cwd=ROOT,env=env,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr)
   calls=[json.loads(x) for x in log.read_text().splitlines()];self.assertEqual(len(calls),12)
   for (check,train),gpus in zip(zip(calls[::2],calls[1::2]),[1,2,4,4,4,8]):
    self.assertEqual(check['env']['CR15_CAPACITY_CHECK'],'1');self.assertEqual(train['env']['CR15_CAPACITY_CHECK'],'0')
    self.assertIn('--gres=gpu:'+str(gpus),train['args']);self.assertIn('--gres=gpu:1',check['args'])
    self.assertNotIn('CR15_MAX_UPDATES',train['env']);self.assertEqual(train['env']['CR15_EPOCH_BUDGET'],'0')
    self.assertEqual(train['env']['CR15_RESUME'],'0');self.assertNotEqual(train['env']['CR15_MODEL'],'GeoTransolver')
    self.assertNotIn('CUDA_VISIBLE_DEVICES',train['env']);self.assertEqual(train['env']['CR15_BATCH_CASES'],'16')
    self.assertTrue(any(a.startswith('--dependency=afterok:') for a in train['args']))
   self.assertTrue((root/'runs/submissions.json').is_file())

if __name__=='__main__':unittest.main()
