"""CPU-only pressure evaluation of saved Rotor37 predictions; no model loading."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def paths(table):
    with table.open() as f:
        return [(table.parent/r['file_path']).resolve() for r in csv.DictReader(f)]


def read(path):
    reader=vtk.vtkXMLUnstructuredGridReader();reader.SetFileName(str(path));reader.Update()
    grid=reader.GetOutput();arr=grid.GetPointData().GetArray('Pressure')
    if arr is None:raise ValueError(f'Missing Pressure: {path}')
    p=vtk_to_numpy(arr).astype(np.float64).reshape(-1)
    offsets=vtk_to_numpy(grid.GetCells().GetOffsetsArray())
    if not np.all(np.diff(offsets)==4) or not np.all(vtk_to_numpy(grid.GetCellTypesArray())==9):
        raise ValueError('Expected quadrilateral surface mesh')
    quads=vtk_to_numpy(grid.GetCells().GetConnectivityArray()).reshape(-1,4).copy()
    if not np.isfinite(p).all():raise ValueError('Nonfinite pressure')
    return p,quads


def ci(values,rng_indices):
    return np.quantile(values[rng_indices].mean(axis=1),[.025,.975]).tolist()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--report',default='reports/transolver_long')
    ap.add_argument('--output',default='reports/transolver_long/pressure_benchmark')
    args=ap.parse_args();root=args.root.resolve();report=root/args.report;out=root/args.output
    train=paths(root/'runs/rotor37/data/processed/train.csv');test=paths(root/'runs/rotor37/data/raw/test.csv')
    valid=paths(root/'runs/rotor37/data/processed/valid.csv')
    assert len(train)==800 and len(test)==100 and len(valid)==100
    assert len(set(train+valid+test))==1000
    p0,quads=read(test[0]);edges=np.unique(np.sort(np.stack((quads,np.roll(quads,-1,axis=1)),axis=-1).reshape(-1,2),axis=1),axis=0)
    assert len(p0)==29773 and len(edges)==59436
    # Stable population variance over all training nodes, matching the official definition.
    count=0;mean=0.;m2=0.
    for index,path in enumerate(train):
        p,q=read(path);assert np.array_equal(q,quads)
        n=len(p);mu=p.mean();total=count+n;delta=mu-mean
        m2+=np.sum((p-mu)**2)+delta**2*count*n/total;mean+=delta*n/total;count=total
        if (index+1)%200==0:print(f'training pressure statistics {index+1}/800',flush=True)
    sigma=float(np.sqrt(m2/count));assert sigma>0
    source={p.name:p for p in test};assert len(source)==len(test)
    targets={}
    for name,path in source.items():
        p,q=read(path);assert np.array_equal(q,quads);targets[name]=p
    rng=np.random.default_rng(42);draws=rng.integers(0,len(test),size=(2000,len(test)))
    rows=[];summaries=[];per_model={};strong_count=int(np.ceil(.01*len(edges)))
    for model in ['GeoTransolver','Transolver_3','POD']:
        with np.load(report/f'{model}_predictions.npz',allow_pickle=False) as data:
            names=data['sample_names'].astype(str);pred=data['pressure_pa'].astype(np.float64)
        assert len(names)==len(set(names))==len(test) and set(names)==set(source)
        assert pred.shape==(100,29773) and np.isfinite(pred).all()
        model_rows=[];sse=0.;truth_squares=0.
        for name,p in zip(names,pred):
            t=targets[name];err=p-t;true_jump=t[edges[:,0]]-t[edges[:,1]]
            jump_error=err[edges[:,0]]-err[edges[:,1]]
            selected=np.argsort(-np.abs(true_jump),kind='stable')[:strong_count]
            row=dict(model=model,sample=name,relative_l2=float(np.linalg.norm(err)/np.linalg.norm(t)),
                     pressure_mse=float(np.mean(err**2)),pressure_mae=float(np.mean(np.abs(err))),
                     jump_mse=float(np.mean((jump_error/sigma)**2)),
                     strongest_jump_mse=float(np.mean((jump_error[selected]/sigma)**2)))
            model_rows.append(row);sse+=float(err@err);truth_squares+=float(t@t)
        model_rows.sort(key=lambda r:r['sample']);rows.extend(model_rows);per_model[model]=model_rows
        rel=np.array([r['relative_l2'] for r in model_rows]);jm=np.array([r['jump_mse'] for r in model_rows]);sm=np.array([r['strongest_jump_mse'] for r in model_rows])
        summary=dict(model=model,cases=len(names),pressure_relative_l2_percent=float(100*rel.mean()),
            pressure_relative_l2_percent_bootstrap_95ci=(100*np.array(ci(rel,draws))).tolist(),
            pressure_relative_l2_percent_median=float(100*np.median(rel)),pressure_relative_l2_percent_p95=float(100*np.quantile(rel,.95)),
            pooled_pressure_relative_l2_percent=float(100*np.sqrt(sse/truth_squares)),
            pressure_rmse=float(np.sqrt(np.mean([r['pressure_mse'] for r in model_rows]))),
            pressure_mae=float(np.mean([r['pressure_mae'] for r in model_rows])),
            jump_rmse_over_training_sigma=float(np.sqrt(jm.mean())),
            strongest_1pct_jump_rmse_over_training_sigma=float(np.sqrt(sm.mean())),
            jump_rmse_bootstrap_95ci=np.sqrt(ci(jm,draws)).tolist(),strongest_jump_rmse_bootstrap_95ci=np.sqrt(ci(sm,draws)).tolist(),
            prediction_sha256=hashlib.sha256((report/f'{model}_predictions.npz').read_bytes()).hexdigest())
        # Cross-check against the historical independent evaluation, allowing saved float32 rounding.
        old=next(r for r in json.loads((report/'comparison.json').read_text())['models'] if r['model']==model)
        assert np.isclose(summary['pressure_rmse'],old['rmse_pa'],rtol=1e-5)
        assert np.isclose(summary['pooled_pressure_relative_l2_percent'],old['relative_l2_percent'],rtol=1e-5)
        summaries.append(summary);print(json.dumps(summary),flush=True)
    paired=[]
    for reference in ['Transolver_3','POD']:
        a=per_model['GeoTransolver'];b=per_model[reference];assert [r['sample'] for r in a]==[r['sample'] for r in b]
        entry=dict(candidate='GeoTransolver',reference=reference)
        for key in ['pressure_mse','jump_mse','strongest_jump_mse']:
            av=np.array([r[key] for r in a]);bv=np.array([r[key] for r in b])
            ratio=np.sqrt(av[draws].sum(axis=1)/bv[draws].sum(axis=1))
            entry[key+'_rmse_ratio_95ci']=np.quantile(ratio,[.025,.975]).tolist()
        paired.append(entry)
    protocol=dict(train_cases=len(train),valid_cases=len(valid),test_cases=len(test),nodes_per_case=len(p0),unique_edges=len(edges),strongest_edges_per_case=strong_count,
        training_pressure_mean=mean,training_pressure_population_std=sigma,
        test_sample_names=sorted(source),train_sample_names=sorted(p.name for p in train),
        relative_l2='mean over cases of norm(prediction-truth)/norm(truth)',
        jump='sqrt(mean over cases and selected edges of squared signed-jump error / train_pressure_std**2)',
        strongest='ceil(0.01*E) largest absolute true jumps per case; stable ties; sorted unique undirected quad edges',
        bootstrap='2000 paired case resamples, numpy default_rng(42); fixed trained checkpoints, not training-seed uncertainty',
        inputs='Historical reference geometry and Omega/P; pressure only',
        limitation='Same metric formulas as NVIDIA, different input/target setup and split algorithm (Python random vs NumPy); test not validation; no SOTA ranking.',
        units='Historical exports label pressure as Pa; upstream official example states source units are undeclared.',
        official_evaluation_source='https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/rotor37_geotransolver/evaluate.py',
        official_statistics_source='https://github.com/NVIDIA/physicsnemo/blob/main/examples/cfd/rotor37_geotransolver/basis.py')
    out.mkdir(parents=True,exist_ok=True)
    (out/'metrics.json').write_text(json.dumps(dict(protocol=protocol,models=summaries,paired_comparisons=paired),indent=2)+'\n')
    for filename,values in [('cases.csv',rows),('metrics.csv',summaries)]:
        with (out/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    lines=['# Rotor37 saved-prediction pressure evaluation','',
           'CPU-only evaluation of the existing 100 held-out cases; no retraining. Training-only pressure standard deviation: '+str(sigma)+'.',
           '', '| Model | Mean case relative L2 (%) | Pooled relative L2 (%) | Pressure RMSE (export scale) | Jump RMSE / sigma | Strongest 1% jump RMSE / sigma |',
           '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['model']} | {r['pressure_relative_l2_percent']:.6f} | {r['pooled_pressure_relative_l2_percent']:.6f} | {r['pressure_rmse']:.4f} | {r['jump_rmse_over_training_sigma']:.6f} | {r['strongest_1pct_jump_rmse_over_training_sigma']:.6f} |")
    lines+=['',protocol['limitation'],'',protocol['units'],'','See metrics.json for definitions, prediction hashes, sample membership, confidence intervals and paired comparisons.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Results:',out)

if __name__=='__main__':main()
