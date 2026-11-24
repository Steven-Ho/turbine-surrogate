import json
import os
import socket
import subprocess
import sys
import unittest
import numpy as np
from tests.test_fann_ddp import DDPTests
from tests.test_fann_formal import ROOT
from scripts.run_fann_geo_scaling import compare_to_reference,multinode_command


class MultinodeScalingTests(unittest.TestCase):
    def test_command_and_reference_metrics(self):
        cmd=multinode_command(2,8,'node0',12345,['scripts/train_fann_ddp.py'])
        self.assertIn('--ntasks=2',cmd)
        self.assertIn('SLURM_PROCID',cmd[7])
        self.assertEqual(cmd[-6:-1],[sys.executable,'2','8','node0','12345'])
        row={'gpus':16,'cases_per_second':.6}
        compare_to_reference(row,{'results':[{'gpus':1,'cases_per_second':.05},{'gpus':8,'cases_per_second':.3}]})
        self.assertAlmostEqual(row['speedup_vs_8gpu'],2.)
        self.assertAlmostEqual(row['efficiency_vs_8gpu'],1.)

    def test_two_agent_cpu_launch(self):
        case=DDPTests();case.setUp()
        try:
            with np.load(case.root/'case_1.npz') as data:sample={key:data[key] for key in data.files}
            sample['coordinates']=sample['coordinates']+3;sample['geometry_id']=np.array([5])
            extra=case.root/'case_5.npz';np.savez_compressed(extra,**sample)
            split=json.loads(case.split.read_text());split['train'].append(str(extra));case.split.write_text(json.dumps(split))
            # Emulate two Slurm agents on localhost, using real torchrun rendezvous.
            stub=case.root/'srun'
            stub.write_text('#!'+sys.executable+'''\nimport os,sys,subprocess,signal,time
args=sys.argv[1:];start=next(i for i,v in enumerate(args) if not v.startswith('--'))
processes=[]
def stop(*unused):
 for p in processes:
  if p.poll() is None:p.terminate()
 raise SystemExit(1)
signal.signal(signal.SIGTERM,stop)
for rank in range(2):processes.append(subprocess.Popen(args[start:],env=dict(os.environ,SLURM_PROCID=str(rank))))
while any(p.poll() is None for p in processes):
 if any(p.poll() not in (None,0) for p in processes):stop()
 time.sleep(.05)
raise SystemExit(max(p.returncode for p in processes))
''');stub.chmod(0o755)
            with socket.socket() as connection:connection.bind(('127.0.0.1',0));port=connection.getsockname()[1]
            env=dict(case.env,CUDA_VISIBLE_DEVICES='',PATH=str(case.root)+':'+os.environ['PATH'],TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'))
            output=case.root/'multi'
            result=subprocess.run([sys.executable,'scripts/run_fann_geo_scaling.py','--splits',str(case.split),
                '--output',str(output),'--nodes','2','--gpu-counts','2','--master-addr','127.0.0.1','--master-port',str(port),
                '--warmup-cases','2','--measure-cases','2','--threads','1','--device','cpu','--backend','gloo'],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=150)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['state'],'complete')
            row=report['results'][0]
            self.assertEqual((row['nodes'],row['gpus_per_node'],row['measured_cases']),(2,1,2))
            self.assertIn('DDP world_size=2',result.stdout)
        finally:case.doCleanups()
