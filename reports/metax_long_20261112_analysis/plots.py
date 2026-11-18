import os
os.environ['MPLBACKEND']='Agg'
import json,csv
from pathlib import Path
import matplotlib.pyplot as plt
p=Path('/tmp/runs-final-analysis/report');data=json.loads((p/'analysis.json').read_text());models=[r['model'] for r in data['models']];rows=list(csv.DictReader((p/'curves.csv').open()))
plt.rcParams.update({'font.size':10,'figure.dpi':130})
fig,axs=plt.subplots(4,2,figsize=(12,14))
for ax,m in zip(axs.flat,models):
 for env,color in [('cluster','#167ab5'),('local','#d87321')]:
  points={int(r['epoch']):float(r['val_loss']) for r in rows if r['model']==m and r['environment']==env};xx=sorted(points)
  ax.plot(xx,[points[x] for x in xx],label=env,color=color,lw=1.3)
 ax.set_title(m);ax.set_yscale('log');ax.set_xlabel('Epoch');ax.set_ylabel('Native validation MSE');ax.grid(alpha=.2);ax.legend()
axs.flat[-1].axis('off');fig.suptitle('Validation curves: compare within each model only',fontsize=15);fig.tight_layout(rect=(0,0,1,.975));fig.savefig(p/'validation_curves.png');plt.close(fig)
stages=[r for r in data['stages'] if r['stage']=='20'];fig,ax=plt.subplots(figsize=(11,5));x=list(range(7));w=.36
ax.bar([v-w/2 for v in x],[r['local20_seconds']/20 for r in stages],w,label='Local RTX 3090')
ax.bar([v+w/2 for v in x],[r['mean_seconds_per_epoch'] for r in stages],w,label='Cluster X203 (Geo optimized)')
ax.set_xticks(x,models,rotation=20);ax.set_ylabel('Seconds / epoch');ax.set_title('First 20 epochs: training-loop time, including validation');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(p/'timing_20epochs.png');plt.close(fig)
sc=[r for r in data['models'] if r['rmse_pa']];sc.sort(key=lambda r:float(r['rmse_pa']));fig,ax=plt.subplots(figsize=(9,4.5));bars=ax.barh([r['model'] for r in sc],[float(r['rmse_pa']) for r in sc]);ax.invert_yaxis()
for bar,r in zip(bars,sc):ax.text(bar.get_width()+200,bar.get_y()+bar.get_height()/2,f"{float(r['rmse_pa']):,.0f}",va='center')
ax.set_xlim(0,29000);ax.set_xlabel('Test RMSE (Pa), lower is better');ax.set_title('Available cluster test results (GCN metrics absent)');fig.tight_layout();fig.savefig(p/'test_rmse.png');plt.close(fig)
