"""Check exclusive fresh 400-epoch LR control without allocating GPUs."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class LR2SubmissionTests(unittest.TestCase):
 def test_fresh_400_epochs_exclusive_and_clean_environment(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);stub=folder/'sbatch';trace=folder/'trace.json'
   stub.write_text('#!'+sys.executable+'\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["TRACE"]).write_text(json.dumps(dict(env=dict(os.environ),args=sys.argv[1:])))\nprint(12345)\n');stub.chmod(0o755)
   run=folder/'fresh'
   env=dict(os.environ,PATH=tmp+':'+os.environ['PATH'],TRACE=str(trace),CR15_LR2_RUN_ROOT=str(run),
    CR15_MODEL='UNet',CR15_RESUME='1',CR15_MAX_UPDATES='20',CR15_EPOCH_BUDGET='100',CR15_EPOCHS='100',
    CR15_CAPACITY_CHECK='1',CR15_RUN_ROOT='old_run',CUDA_VISIBLE_DEVICES='0',CR15_PREPARED='/cluster/prepared')
   result=subprocess.run(['bash','scripts/submit_cr15_geo_lr2_long.sh'],cwd=ROOT,env=env,text=True,capture_output=True)
   self.assertEqual(result.returncode,0,result.stderr)
   report=json.loads(trace.read_text());e=report['env'];args=report['args']
   for value in ['--exclusive','--nodes=2','--ntasks-per-node=1','--gres=gpu:8','--cpus-per-task=32','--mem=64G']:
    self.assertIn(value,args)
   expected=dict(CR15_RUN_ROOT=str(run),CR15_MODEL='GeoTransolver',CR15_SCALE='base',CR15_AMP='bf16',
    CR15_BATCH_CASES='16',CR15_GPUS_PER_NODE='8',CR15_EPOCHS='400',CR15_EPOCH_BUDGET='0',CR15_RESUME='0',
    CR15_CAPACITY_CHECK='0',CR15_LARGEST_FIRST='0',CR15_LEARNING_RATE='0.0002',
    CR15_LR_PATIENCE='20',CR15_LR_THRESHOLD='0.001',CR15_LR_COOLDOWN='5',CR15_PREPARED='/cluster/prepared')
   for key,value in expected.items():self.assertEqual(e[key],value)
   self.assertNotIn('CR15_MAX_UPDATES',e);self.assertNotIn('CUDA_VISIBLE_DEVICES',e)
   self.assertIn('--output=logs/cr15_geo_lr2_long_%j.out',args)
   # Existing weights must never be overwritten by this fresh-run entry point.
   (run/'GeoTransolver').mkdir(parents=True);(run/'GeoTransolver/last.pt').touch();trace.unlink()
   failed=subprocess.run(['bash','scripts/submit_cr15_geo_lr2_long.sh'],cwd=ROOT,env=env,text=True,capture_output=True)
   self.assertNotEqual(failed.returncode,0);self.assertFalse(trace.exists())

if __name__=='__main__':unittest.main()
