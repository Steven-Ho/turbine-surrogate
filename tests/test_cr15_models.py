"""CR15 six-model full-field evaluation, graph exactness, DDP and resume checks."""
import csv,json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from test_cr15_training import fixture as base_fixture
ROOT=Path(__file__).resolve().parents[1]
def fixture(root):
 prepared=base_fixture(root);splits=json.loads((prepared/'splits.json').read_text());coords=[];targets=[]
 for index,(split,paths) in enumerate(splits.items()):
  for case,path in enumerate(paths):
   file=prepared/path
   with np.load(file) as d:sample={k:d[k] for k in d.files}
   ni=3+case;xx,yy,zz=np.meshgrid(np.arange(ni),np.arange(3),np.arange(3),indexing='ij')
   xyz=np.column_stack([v.reshape(-1,order='F') for v in [xx,yy,zz]]).astype('float32')*.02+.4+case*.003
   t=np.column_stack([xyz[:,0]*10+case,xyz[:,1]*20-case,xyz[:,2]*30,1e6+xyz[:,0]*1e5+case*1e4,1800+xyz[:,1]*10+case]).astype('float32')
   sample.update(coordinates=xyz,targets=t,original_node_ids=np.arange(len(xyz)),block_shapes=np.array([[ni,3,3]]),block_offsets=np.array([0,len(xyz)]))
   np.savez(file,**sample)
   meta=prepared/'metadata'/(file.stem+'.json');v=json.loads(meta.read_text());v['nodes']=len(xyz);meta.write_text(json.dumps(v))
   if split=='train':coords.append(xyz);targets.append(t)
 values=np.concatenate(targets).astype('float64');xyz=np.concatenate(coords)
 stats=dict(source_split='train',nodes=len(values),mean=values.mean(0).tolist(),std=values.std(0).tolist(),coordinate_minimum=xyz.min(0).tolist(),coordinate_maximum=xyz.max(0).tolist())
 (prepared/'train_statistics.json').write_text(json.dumps(stats));return prepared

class SixModelTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.root=Path(self.tmp.name);self.prepared=fixture(self.root)
  self.env=dict(os.environ,PYTHONPATH=str(ROOT),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLCONFIGDIR=str(self.root/'mpl'),TURBINE_GEO_QUERY_BACKEND='kdtree',TURBINE_GEO_CACHE_DIR=str(self.root/'geo'))
 def run_model(self,name,output,extra=(),ddp=False):
  cmd=[sys.executable]
  if ddp:cmd+=['-m','torch.distributed.run','--standalone','--nproc_per_node=2']
  cmd+=['scripts/train_cr15.py','--prepared',str(self.prepared),'--output',str(output),'--model',name,'--device','cpu','--scale','small','--amp','off','--grid','32','--pod-rank','2','--geometry-bins','4','--graph-owner-nodes','10','--graph-edge-chunk','32','--model-cache',str(self.root/'cache'),'--epochs','2','--batch-cases','2','--chunk','16','--threads','1','--checkpoint-every','1',*extra]
  result=subprocess.run(cmd,cwd=ROOT,env=self.env,text=True,capture_output=True,timeout=240)
  self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
 def test_six_full_train_and_independent_physical_metrics(self):
  for name in ['POD','UNet','GCN','Graph_UNet','Graph_Transformer','Transolver_3']:
   with self.subTest(model=name):
    self.run_model(name,self.root/'all',['--epochs','1'])
    report=json.loads((self.root/'all'/name/'report.json').read_text())
    self.assertEqual(report['model'],name);self.assertEqual(len(report['test']['rmse']),5)
    self.assertIn('baseline',report)
    for row in report['test']['samples']:
     with np.load(row['prediction_file']) as p,np.load(row['source']) as t:
      np.testing.assert_allclose(np.sqrt(((p['prediction'].astype(float)-t['targets'])**2).mean(0)),report['test']['rmse'],rtol=1e-9)
      np.testing.assert_array_equal(p['original_node_ids'],t['original_node_ids'])
 def test_pod_exact_resume(self):
  self.run_model('POD',self.root/'full')
  self.run_model('POD',self.root/'resume',['--epoch-budget','1'])
  self.assertFalse((self.root/'resume/POD/report.json').exists())
  self.run_model('POD',self.root/'resume',['--resume'])
  a=torch.load(self.root/'full/POD/last.pt',weights_only=False);b=torch.load(self.root/'resume/POD/last.pt',weights_only=False)
  self.assertEqual(a['history'],b['history'])
  for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
 def test_gcn_ddp_tail_and_accumulation(self):
  self.run_model('GCN',self.root/'single')
  self.run_model('GCN',self.root/'ddp',ddp=True)
  a=torch.load(self.root/'single/GCN/last.pt',weights_only=False);b=torch.load(self.root/'ddp/GCN/last.pt',weights_only=False)
  self.assertEqual(a['global_step'],4);self.assertEqual(b['global_step'],4)
  for key,state in a['optimizer_state_dict']['state'].items():
   torch.testing.assert_close(state['exp_avg'],b['optimizer_state_dict']['state'][key]['exp_avg'],rtol=3e-3,atol=3e-6)
 def test_graph_owner_halo_matches_full_graph(self):
  from tfai.benchmarks.cr15 import load_prepared,CR15Normalizer,CR15Store
  from tfai.benchmarks.cr15_models import make_model,model_input
  from torch_geometric.data import Data
  prepared=load_prepared(self.prepared);norm=CR15Normalizer(prepared);sample=CR15Store(prepared['paths']['train'])[0]
  config=dict(scale='small',amp='off',chunk=16,offload=False,model_cache=str(self.root/'cache'),graph_owner_nodes=10,graph_edge_chunk=32)
  torch.set_num_threads(1)
  for name in ['GCN','Graph_Transformer','Graph_UNet']:
   with self.subTest(model=name):
    config['model']=name;model=make_model(prepared,config).eval()
    data=model_input(model,norm,sample,torch.device('cpu'));a=model.executor.adjacency(sample).tocoo()
    edges=torch.tensor(np.stack([a.col,a.row]),dtype=torch.long);delta=data.x[edges[1],:3]-data.x[edges[0],:3]
    graph=Data(x=data.x,edge_index=edges,edge_attr=torch.cat([delta,delta.norm(dim=1,keepdim=True)],1))
    with torch.no_grad():expected=model.net(graph);actual=model(data)
    torch.testing.assert_close(actual,expected,rtol=5e-4,atol=2e-6)

if __name__=='__main__':unittest.main()
