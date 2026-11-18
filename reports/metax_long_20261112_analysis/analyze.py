import csv,json,re,statistics,hashlib,datetime
from pathlib import Path
BASE=Path('/tmp/runs-final-analysis');REPO=Path('/data/hesc/Projects/turbine-surrogate');OUT=BASE/'report';OUT.mkdir(exist_ok=True)
OLD=BASE/'metax_long_20261111_100428_4166078_logs/runs/metax_long_20261111_100428_4166078';NEW=BASE/'metax_geo_kdtree_1455_logs/runs/metax_geo_kdtree_1455'
MODELS=['POD','UNet','GCN','Graph_UNet','Graph_Transformer','Transolver_3','GeoTransolver']
pattern=re.compile(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d).*验证周期完成: 验证周期=(\d+)/(\d+) global_step=(\d+) data_epoch=(\d+) train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)')
def curve(path):
 if not path.exists():return []
 rows=[]
 for m in pattern.finditer(path.read_text(errors='replace')):
  t,e,target,step,de,tr,va=m.groups();rows.append(dict(time=t,epoch=int(e),target=int(target),step=int(step),train_loss=float(tr),val_loss=float(va)))
 return rows

def writecsv(name,rows):
 if not rows:return
 keys=list(dict.fromkeys(k for r in rows for k in r));f=(OUT/name).open('w');w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows);f.close()
metrics=list(csv.DictReader((OLD/'metrics_original_nodes.csv').open()));scores={r['model']:r for r in metrics}
local20={r['model']:r for r in csv.DictReader((REPO/'reports/comparison20/metrics.csv').open())}
local20.update({r['model']:r for r in csv.DictReader((REPO/'reports/transolver_comparison20/metrics.csv').open())} if (REPO/'reports/transolver_comparison20/metrics.csv').exists() else {})
local_long={r['model']:r for r in csv.DictReader((REPO/'reports/transolver_long/metrics.csv').open())}
baseline=json.loads((NEW/'GeoTransolver/local_baseline.json').read_text())['models']
summaries=[];stages=[];allcurves=[];audit=[]
for model in MODELS:
 run=NEW/model if model=='GeoTransolver' else OLD/model
 st=json.loads((run/'status.json').read_text());split=json.loads((run/'data_split.json').read_text());b=baseline[model]
 names=b['split_sample_names'];audit.append(dict(model=model,pool_matches_local=set(split['pool'])==set(names['train']+names['valid']),test_matches_local=set(split['test'])=={r['sample']+'.vtu' for r in csv.DictReader((REPO/'reports/comparison20/per_sample_metrics.csv').open()) if r['model']=='POD'},seed=split['seed']))
 aggregate=[];total=0.;last=0.
 for stage in ['20','200','400']:
  f=run/f'train_{stage}_console.log'
  if not f.exists():continue
  rows=curve(f);aggregate+=rows
  info=st['stages'].get(stage,{})
  localpath=None
  if stage=='20':
   localpath=REPO/(b['twenty_epoch']['run_directory'])/'train_console.log'
  elif model in local_long:localpath=REPO/f'runs/transolver_long/{model}/train_{stage}_console.log'
  lrows=curve(localpath) if localpath else []
  intervals=[(datetime.datetime.fromisoformat(y['time'])-datetime.datetime.fromisoformat(x['time'])).total_seconds()/(y['epoch']-x['epoch']) for x,y in zip(rows,rows[1:]) if y['epoch']>x['epoch']]
  lint=[(datetime.datetime.fromisoformat(y['time'])-datetime.datetime.fromisoformat(x['time'])).total_seconds()/(y['epoch']-x['epoch']) for x,y in zip(lrows,lrows[1:]) if y['epoch']>x['epoch']]
  sec=info.get('training_seconds');epochs=info.get('completed_epochs',rows[-1]['epoch'] if rows else 0)
  add=epochs-last;last=epochs
  record=dict(model=model,stage=stage,completed_epochs=epochs,additional_epochs=add,training_seconds=sec,pipeline_seconds=info.get('pipeline_train_seconds'),mean_seconds_per_epoch=sec/add if sec and add else None,cycle_median_seconds=statistics.median(intervals) if intervals else None,local_cycle_median_seconds=statistics.median(lint) if lint else None,local20_seconds=b['twenty_epoch']['training_seconds'] if stage=='20' else None,cluster_over_local20_ratio=sec/b['twenty_epoch']['training_seconds'] if stage=='20' and sec else None)
  if model=='Transolver_3' and stage=='200':record['note']='Local log resumed; 784.49 sec covers only last segment. Use cycle median, not whole-stage ratio.'
  stages.append(record)
  for env,rr in [('cluster',rows),('local',lrows)]:
   for r in rr:allcurves.append(dict(model=model,environment=env,stage=stage,**r))
  if sec:total+=sec
 valid={r['epoch']:r for r in aggregate};vals=list(valid.values());best=min(vals,key=lambda r:r['val_loss'])
 score=scores.get(model,{})
 summary=dict(model=model,state=st['state'],latest_validated_epoch=max(valid),best_logged_validation_epoch=best['epoch'],best_validation_loss=best['val_loss'],last_train_loss=vals[-1]['train_loss'],last_val_loss=vals[-1]['val_loss'],training_seconds_total_completed_stages=total,task_seconds=st.get('task_seconds'),test_seconds=st.get('pipeline_test_seconds'),rmse_pa=score.get('rmse_pa'),mape_percent=score.get('mape_percent'),r2=score.get('r2'),test_best_epoch=float(score['best_optimizer_steps'])/800 if score else None,local_long_rmse=local_long.get(model,{}).get('rmse_pa'))
 if model in scores and model in local_long:summary['rmse_change_vs_local_long_percent']=(float(score['rmse_pa'])/float(local_long[model]['rmse_pa'])-1)*100
 summaries.append(summary)
 writecsv(model+'_cluster_history.csv',vals)
 # scan canonical logs only, no duplicated hydra output
 issues=[]
 for f in sorted(run.glob('train_*_console.log')):
  text=f.read_text(errors='replace')
  nonfinite=re.findall(r'(?:batch_loss|train_loss|val_loss)=(?:nan|[-+]?inf)\b',text,re.I)
  issues.append(dict(file=f.name,nonfinite_count=len(nonfinite),traceback=text.count('Traceback (most recent call last)'),error_lines=sum(' - ERROR - ' in l for l in text.splitlines())))
 summary['training_log_audit']=issues
writecsv('model_summary.csv',[{k:v for k,v in r.items() if k!='training_log_audit'} for r in summaries]);writecsv('stage_timing.csv',stages);writecsv('curves.csv',allcurves);writecsv('split_audit.csv',audit)
(OUT/'analysis.json').write_text(json.dumps(dict(models=summaries,stages=stages,split_audit=audit,source_zip_sha256=hashlib.sha256(Path('/data/hesc/runs.zip').read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(summaries,indent=2));print('SPLITS',audit)
