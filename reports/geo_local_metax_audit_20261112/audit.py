from pathlib import Path
import csv,json,re,hashlib
import torch,yaml
torch.set_num_threads(1)
R=Path('/data/hesc/Projects/turbine-surrogate');C=Path('/tmp/runs-final-analysis/metax_geo_kdtree_1455_logs/runs/metax_geo_kdtree_1455/GeoTransolver');O=Path('/tmp/geo-audit')
d=torch.load(R/'runs/transolver_long/GeoTransolver/train/best_model.pt',map_location='cpu',weights_only=False)
local=d['config'];cluster=yaml.safe_load((C/'configs/stage_400.yaml').read_text())
def flatten(v,p=''):
 if isinstance(v,dict):return {kk:vv for k,x in v.items() for kk,vv in flatten(x,p+'.'+k if p else k).items()}
 return {p:v}
a,b=flatten(local),flatten(cluster);diffs=[{'key':k,'local':a.get(k),'cluster':b.get(k)} for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
(O/'config_differences.json').write_text(json.dumps(diffs,indent=2,ensure_ascii=False)+'\n');print('CONFIG DIFF',json.dumps(diffs,ensure_ascii=False))
(O/'local_config.json').write_text(json.dumps(local,indent=2,ensure_ascii=False)+'\n')
(O/'local_norm_params.json').write_text(json.dumps(d['norm_params'],indent=2)+'\n')
tables={}
for split in ['train','valid']:
 rows=list(csv.DictReader((R/f'runs/transolver_long/GeoTransolver/data/processed/effective_{split}.csv').open()));names=[Path(r['file_path']).name for r in rows];tables[split]=names
pool=json.loads((C/'data_split.json').read_text());print('pool membership',set(pool['pool'])==set(tables['train']+tables['valid']))
(O/'local_effective_split.json').write_text(json.dumps(tables,indent=2)+'\n')
rows=[]
for env in ['local','cluster']:
 byepoch={}
 for stage in ['20','200','400']:
  f=(R/'runs/transolver_comparison20/GeoTransolver/train_console.log' if stage=='20' else R/f'runs/transolver_long/GeoTransolver/train_{stage}_console.log') if env=='local' else C/f'train_{stage}_console.log'
  for step,lr in re.findall(r'学习率调度: step=(\d+) lr=\[([^\]]+)\]',f.read_text()):byepoch[int(step)//800]=float(lr)
 prev=.0001
 events=[]
 for ep,lr in sorted(byepoch.items()):
  rows.append({'environment':env,'epoch':ep,'lr_after_validation':lr})
  if lr!=prev:events.append({'epoch':ep,'from':prev,'to':lr});prev=lr
 print(env,'LR EVENTS',events);(O/f'{env}_lr_events.json').write_text(json.dumps(events,indent=2)+'\n')
with (O/'learning_rate_by_epoch.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
localcases=[r for r in csv.DictReader((R/'reports/transolver_long/per_sample_metrics.csv').open()) if r['model']=='GeoTransolver']
with (O/'local_per_sample_metrics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(localcases[0]));w.writeheader();w.writerows(localcases)
print('checkpoint val_loss',d['val_loss'],'step',d['global_step'])
