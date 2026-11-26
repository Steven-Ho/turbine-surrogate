"""CPU-only, bounded-case faNN report; imports no Torch, VTK or model code.

Full-domain metrics come from report.json. Visualizations use deterministic
subsamples of a geometric slab, never interpolate through holes in the mesh.
"""
import argparse
import csv
import html
import json
from pathlib import Path
from datetime import datetime
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FIELDS=['VelocityX','VelocityY','VelocityZ','Pressure','Temperature','Density']
UNITS=['m/s','m/s','m/s','Pa','K','kg/m^3']


def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def json_write(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def discover(path):
    if path is not None:
        folder=Path(path)
        if (folder/'GeoTransolver/report.json').exists():folder=folder/'GeoTransolver'
        if not (folder/'report.json').is_file():raise ValueError(f'Missing report.json in {folder}')
        return folder
    found=sorted(p.parent for p in Path('runs').glob('fann_geo_domain4_8nodes_*/GeoTransolver/report.json'))
    if len(found)!=1:raise ValueError('Specify --run-root: expected exactly one completed domain run')
    return found[0]


def locate(value, folder, mappings, prediction=False):
    text=str(value)
    for old,new in mappings:
        if text==old or text.startswith(old.rstrip('/')+'/'):
            text=new.rstrip('/')+text[len(old.rstrip('/')):];break
    path=Path(text)
    if path.is_file():return path
    if not path.is_absolute():
        for base in [folder,Path.cwd()]:
            if (base/path).is_file():return base/path
    if prediction and '/predictions/' in text:
        candidate=folder/'predictions'/text.split('/predictions/',1)[1]
        if candidate.is_file():return candidate
    raise FileNotFoundError(f'{value}: missing; use --path-map OLD=NEW if directories moved')


def source_arrays(path):
    # Select NPZ members, avoiding cells/connectivity and other large arrays.
    with np.load(path,allow_pickle=False) as data:
        xyz=data['coordinates'];truth=data['targets']
        if 'original_node_ids' in data.files:ids=data['original_node_ids']
        elif 'mesh_file' in data.files:
            mesh=Path(str(data['mesh_file']))
            if not mesh.is_absolute():mesh=path.parent/mesh
            with np.load(mesh,allow_pickle=False) as mesh_data:ids=mesh_data['original_node_ids']
        else:raise ValueError(f'{path}: no original node IDs')
    return xyz,truth,ids


def training_plots(history, out):
    if not history:return []
    x=np.array([r['epoch'] for r in history]);train=[r['train_loss'] for r in history];val=[r['val_loss'] for r in history]
    best=int(np.argmin(val))
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    axes[0].plot(x,train,label='Train');axes[0].plot(x,val,label='Validation',alpha=.8)
    axes[0].scatter(x[best],val[best],color='green',s=25,zorder=5,label=f'Best: {val[best]:.6g}, epoch {x[best]}')
    axes[0].set(xlabel='Epoch',ylabel='Normalized MSE',title='Training / validation');axes[0].legend(fontsize=8)
    axes[1].step(x,[r['lr'] for r in history],where='post');axes[1].set_yscale('log')
    axes[1].set(xlabel='Epoch',ylabel='Learning rate',title='Learning rate schedule')
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(out/'training.png',dpi=150);fig.savefig(out/'training.pdf');plt.close(fig)
    rows=[dict(epoch=r['epoch'],step=r.get('global_step'),train_mse=r['train_loss'],val_mse=r['val_loss'],lr=r['lr']) for r in history]
    write_csv(out/'training.csv',rows)
    images=['training.png']
    if all(len(r.get('validation_rmse',[]))==6 for r in history):
        values=np.asarray([r['validation_rmse'] for r in history])
        fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
        for i,ax in enumerate(axes.flat):
            ax.plot(x,values[:,i]);ax.set(title=FIELDS[i],xlabel='Epoch',ylabel=f'Validation RMSE ({UNITS[i]})');ax.grid(alpha=.2)
        fig.savefig(out/'field_validation_rmse.png',dpi=150);plt.close(fig);images.append('field_validation_rmse.png')
    return images


def representatives(samples, global_rmse, number):
    scale=np.maximum(np.asarray(global_rmse,dtype=float),1e-30)
    scored=sorted([(float(np.sqrt(np.mean((np.asarray(row['rmse'])/scale)**2))),row) for row in samples],key=lambda item:(item[0],item[1]['source']))
    positions=np.unique(np.rint(np.linspace(0,len(scored)-1,min(number,len(scored)))).astype(int))
    return [scored[int(i)] for i in positions]


def sample_indices(count, maximum, rng):
    return np.sort(rng.choice(count,size=min(count,maximum),replace=False))


def visualize_case(row, score, folder, out, index, args, mappings):
    prediction_path=locate(row['prediction_file'],folder,mappings,prediction=True)
    source=locate(row['source'],folder,mappings)
    print(f'Case {index}: reading {source}',flush=True)
    xyz,truth,ids=source_arrays(source)
    with np.load(prediction_path,allow_pickle=False) as data:
        predicted=data['prediction'];prediction_ids=data['original_node_ids'];names=data['output_names'].tolist()
    if names!=FIELDS or truth.shape!=predicted.shape or truth.shape!=(len(xyz),6):raise ValueError('Field/order/shape mismatch')
    if not np.array_equal(ids,prediction_ids):raise ValueError('Prediction/truth original node IDs differ; refusing misaligned plots')
    # Validate only this selected pair; no dataset scan or file fingerprint pass.
    rng=np.random.default_rng(args.seed+index)
    point_ids=sample_indices(len(xyz),args.max_points,rng)
    axis='xyz'.index(args.slice_axis);other=[i for i in range(3) if i!=axis]
    coordinate=xyz[:,axis];lo=float(coordinate.min());hi=float(coordinate.max())
    center=lo+args.slice_fraction*(hi-lo);half=args.slice_width*max(hi-lo,1e-12)/2
    distance=np.abs(coordinate-center);slab=np.flatnonzero(distance<=half)
    nearest=False
    if len(slab)<min(200,len(xyz)):
        count=min(2000,len(xyz));slab=np.argpartition(distance,count-1)[:count];nearest=True
    slab=np.sort(slab[sample_indices(len(slab),args.slice_points,rng)])
    compact=dict(coordinates=xyz[point_ids],truth=truth[point_ids],prediction=predicted[point_ids],
                 original_node_ids=ids[point_ids],output_names=np.array(FIELDS),
                 slice_coordinates=xyz[slab],slice_truth=truth[slab],slice_prediction=predicted[slab],
                 slice_original_node_ids=ids[slab])
    if not all(np.isfinite(compact[key]).all() for key in ['coordinates','truth','prediction']):raise ValueError('Non-finite sampled values')
    np.savez_compressed(out/f'case_{index:02d}_sample.npz',**compact)
    images=[];prefix=f'case_{index:02d}'
    for field in args.fields:
        j=FIELDS.index(field);true=truth[slab,j];pred=predicted[slab,j];error=pred-true
        vmin,vmax=np.percentile(np.r_[true,pred],[1,99]);extent=float(np.percentile(np.abs(error),99))
        if vmax<=vmin:vmax=vmin+max(abs(vmin)*1e-6,1e-6)
        extent=max(extent,1e-12)
        fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
        for ax,values,title,cmap,lower,upper in zip(axes,[true,pred,error],['Truth','Prediction','Prediction − truth'],['viridis','viridis','coolwarm'],[vmin,vmin,-extent],[vmax,vmax,extent]):
            plot=ax.scatter(xyz[slab,other[0]],xyz[slab,other[1]],c=values,s=2,linewidths=0,cmap=cmap,vmin=lower,vmax=upper,rasterized=True)
            ax.set(title=title,xlabel='xyz'[other[0]]+' (m)',ylabel='xyz'[other[1]]+' (m)');ax.set_aspect('equal',adjustable='box')
            fig.colorbar(plot,ax=ax,shrink=.8,label=UNITS[j],extend='both')
        fig.suptitle(f'{prefix} | {field} | {args.slice_axis}≈{center:.5g} m | full-case RMSE={row["rmse"][j]:.5g} {UNITS[j]}',fontsize=11)
        name=f'{prefix}_{field}.png';fig.savefig(out/name,dpi=140);plt.close(fig);images.append(name)
    fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
    for j,ax in enumerate(axes.flat):
        a=compact['truth'][:,j];b=compact['prediction'][:,j]
        ax.hexbin(a,b,gridsize=45,mincnt=1,bins='log',cmap='viridis')
        lo=min(float(a.min()),float(b.min()));hi=max(float(a.max()),float(b.max()))
        ax.plot([lo,hi],[lo,hi],'r--',lw=.8)
        ax.set(title=FIELDS[j],xlabel=f'Truth ({UNITS[j]})',ylabel=f'Prediction ({UNITS[j]})')
    name=f'{prefix}_parity.png';fig.savefig(out/name,dpi=140);plt.close(fig);images.append(name)
    metadata=dict(case=index,source=str(source),prediction_file=str(prediction_path),nodes=len(xyz),
        relative_case_score=score,full_case_rmse=row['rmse'],full_case_mae=row.get('mae'),
        sampled_nodes=len(point_ids),slice_nodes=len(slab),slice_axis=args.slice_axis,
        slice_center_m=center,actual_max_distance_to_plane_m=float(distance[slab].max()),
        nearest_nodes_fallback=nearest,images=images)
    print(f'Case {index}: {len(xyz)} nodes -> {len(point_ids)} sampled / {len(slab)} slab nodes',flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--split',choices=['test','validation'],default='test')
    parser.add_argument('--cases',type=int,default=3)
    parser.add_argument('--max-points',type=int,default=30000)
    parser.add_argument('--slice-points',type=int,default=15000)
    parser.add_argument('--slice-axis',choices=list('xyz'),default='z')
    parser.add_argument('--slice-fraction',type=float,default=.5)
    parser.add_argument('--slice-width',type=float,default=.01)
    parser.add_argument('--fields',nargs='+',choices=FIELDS,default=FIELDS)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--path-map',action='append',default=[],metavar='OLD=NEW')
    args=parser.parse_args()
    if min(args.cases,args.max_points,args.slice_points)<1:parser.error('Counts must be positive')
    if not 0<=args.slice_fraction<=1 or not 0<args.slice_width<=1:parser.error('Invalid slab parameters')
    mappings=[]
    for item in args.path_map:
        if '=' not in item:parser.error('path-map must be OLD=NEW')
        mappings.append(item.split('=',1))
    folder=discover(args.run_root)
    output=args.output or folder/('visual_summary_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True,exist_ok=False)
    report=json.loads((folder/'report.json').read_text())
    if report.get('state')!='complete':raise ValueError('Only a completed report can be summarized')
    if report.get('fields')!=FIELDS:raise ValueError('Unexpected report fields')
    history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').is_file() else []
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    images=training_plots(history,output)
    metrics=[]
    for split in ['validation','test']:
        metric=report[split]
        if metric['fields']!=FIELDS:raise ValueError('Unexpected metric fields')
        for j,field in enumerate(FIELDS):
            near=metric.get('near_wall',{}).get('rmse')
            metrics.append(dict(split=split,field=field,unit=UNITS[j],rmse=metric['rmse'][j],
                mae=metric['mae'][j],r2=metric['r2'][j],near_wall_rmse=near[j] if near else None,
                nodes=metric['nodes'],cases=metric['cases']))
    write_csv(output/'field_metrics.csv',metrics)
    samples=report[args.split]['samples']
    if not samples:raise ValueError('No predicted cases in report')
    selected=representatives(samples,report[args.split]['rmse'],args.cases)
    cases=[visualize_case(row,score,folder,output,i,args,mappings) for i,(score,row) in enumerate(selected)]
    summary=dict(model=report['model'],completed_epochs=report['completed_epochs'],best_epoch=report['best_epoch'],
        validation_normalized_mse=report['validation']['normalized_mse'],test_normalized_mse=report['test']['normalized_mse'],
        training_seconds=report.get('training_seconds'),metrics=metrics,cases=cases,
        notes=['Full-domain metrics are copied from report.json, not estimated from plotted samples.',
               'Representative cases span low/median/high relative RMSE scores; each field RMSE is divided by the selected split full-domain field RMSE before ranking. Selection is descriptive only.',
               'Slab plots show original node values with no interpolation; truth/prediction share a 1st–99th percentile color range; signed errors use symmetric 99th-percentile absolute limits.',
               'Sample NPZs contain only the exported subset and cannot replace full-mesh evaluation.'])
    json_write(output/'summary.json',summary)
    json_write(output/'original_report.json',report)
    esc=html.escape
    rows=''.join('<tr>'+''.join(f'<td>{esc(str(r[k]))}</td>' for k in ['split','field','unit','rmse','mae','r2','near_wall_rmse'])+'</tr>' for r in metrics)
    document='<!doctype html><meta charset="utf-8"><title>faNN prediction summary</title><style>body{max-width:1300px;margin:30px auto;font:15px sans-serif;color:#18212b}img{width:100%}td,th{padding:7px;border-bottom:1px solid #ddd;text-align:right}h2{margin-top:35px}</style>'
    document+=f'<h1>{esc(report["model"])} · faNN</h1><p>Completed epochs: {report["completed_epochs"]}; best checkpoint: epoch {report["best_epoch"]}. Validation MSE: {summary["validation_normalized_mse"]:.6g}; test MSE: {summary["test_normalized_mse"]:.6g}.</p>'
    document+='<p>Full-domain physical metrics; RMSE/MAE have the units listed. Plots use node subsets.</p><table><tr>'+''.join(f'<th>{x}</th>' for x in ['Split','Field','Unit','RMSE','MAE','R²','Near-wall RMSE'])+'</tr>'+rows+'</table>'
    for name in images:document+=f'<img src="{name}" alt="{name}">'
    for case in cases:
        document+=f'<h2>Case {case["case"]}: {esc(Path(case["source"]).name)}</h2><p>{case["nodes"]} full nodes; {case["slice_nodes"]} slab nodes; nearest-node fallback: {case["nearest_nodes_fallback"]}.</p>'
        for name in case['images']:document+=f'<img loading="lazy" src="{name}" alt="{name}">'
    document+='<h2>Plot conventions</h2><ul>'+''.join(f'<li>{esc(note)}</li>' for note in summary['notes'])+'</ul>'
    (output/'index.html').write_text(document)
    archive=output.with_name(output.name+'.zip')
    if archive.exists():raise FileExistsError(archive)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.iterdir()):
            if path.is_file():z.write(path,arcname=output.name+'/'+path.name)
    print(f'Done: {output}/index.html\nDownload only: {archive} ({archive.stat().st_size/1024**2:.1f} MiB)',flush=True)


if __name__=='__main__':main()
