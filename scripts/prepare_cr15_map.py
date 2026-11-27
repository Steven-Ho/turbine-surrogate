"""Prepare A-grade CR15 MAP caches using a completed audit; CPU/NumPy only.

No source-file SHA256 pass, CFD solve, model training, inferred density, node
filtering or re-splitting. The manifest's solved outputs are diagnostics only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.audit_cr15_map import record

SPLITS={'train':'train','validation':'valid','test':'test','baseline':'baseline'}


def atomic_json(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,suffix='.json',delete=False) as f:
        temporary=Path(f.name);json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    fields=fields or (list(rows[0]) if rows else ['sample'])
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def safe_id(value):
    if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in value):
        raise ValueError('Unsupported sample ID: '+value)
    return value


def load_contract(path):
    c=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if c['target_raw_indices']!=[1,2,3,0,4] or c['target_names']!=['VelocityX','VelocityY','VelocityZ','Pressure','Temperature']:
        raise ValueError('Unexpected five-field mapping')
    if len(c['raw_field_order'])!=8 or not (len(c['boundary_names'])==len(c['boundary_values'])==len(c['boundary_units'])):
        raise ValueError('Invalid field/boundary contract')
    if c['coordinate_multiplier']<=0 or not np.isfinite(c['boundary_values']).all():raise ValueError('Invalid units/boundaries')
    return c


def selection(root, audit=None):
    rows=read_csv(root/'dataset_manifest.csv')
    names=[safe_id(r['sample']) for r in rows]
    if not rows or len(set(names))!=len(names):raise ValueError('Empty/duplicate sample manifest')
    if any(r['split'] not in SPLITS or r['level'] not in ['A','B'] for r in rows):raise ValueError('Unknown split/grade')
    eligible=[r for r in rows if r['level']=='A'];excluded=[r for r in rows if r['level']!='A']
    audit_rows={}
    if audit is not None:
        summary=json.loads((audit/'summary.json').read_text(encoding='utf-8-sig'))
        if summary['total']!=len(rows) or summary['passed']!=len(rows) or summary['failed']!=0:
            raise ValueError('Expected completed all-passed audit for the manifest')
        raw=read_csv(audit/'eligible_A.csv');audit_rows={r['sample']:r for r in raw}
        if len(raw)!=len(audit_rows) or set(audit_rows)!={r['sample'] for r in eligible}:
            raise ValueError('Audit A-grade membership differs from manifest')
        for r in eligible:
            if audit_rows[r['sample']]['split']!=r['split']:raise ValueError('Audit split differs for '+r['sample'])
    return rows,eligible,excluded,audit_rows


def map_path(value, root, mappings):
    text=str(value).replace('\\','/')
    for old,new in mappings:
        old=old.replace('\\','/').rstrip('/')
        if text==old or text.startswith(old+'/'):
            text=str(new).rstrip('/')+text[len(old):];break
    path=Path(text)
    if not path.is_absolute():path=root/path
    return path


def read_map(path, grid):
    """Read channel-major records; flattened i changes fastest (Fortran order)."""
    blocks=[];flags=[];shapes=[]
    with Path(path).open('rb') as f:
        marker=f.read(4)
        if marker not in [b'\x04\x00\x00\x00',b'\x00\x00\x00\x04']:raise ValueError('Unsupported MAP byte order/header')
        endian='<' if marker==b'\x04\x00\x00\x00' else '>';f.seek(0)
        header=record(f,endian)
        if len(header)!=4:raise ValueError('Invalid block header')
        count=struct.unpack(endian+'i',header)[0]
        if not 1<=count<=10000:raise ValueError('Invalid block count')
        width=3 if grid else 4
        dims=np.frombuffer(record(f,endian),dtype=endian+'i4')
        if dims.size!=count*width:raise ValueError('Dimension record mismatch')
        dims=dims.reshape(count,width)
        for dimension in dims:
            shape=tuple(map(int,dimension[:3]));channels=3 if grid else int(dimension[3])
            if min(shape)<=0 or channels<1 or (not grid and channels!=8):raise ValueError('Invalid block shape/field count')
            n=math.prod(shape);raw=record(f,endian);size=n*channels*8
            if len(raw)!=size and not(grid and len(raw)==size+n*4):raise ValueError('Unsupported MAP data record')
            values=np.frombuffer(raw,dtype=endian+'f8',count=n*channels).reshape(channels,n).T.copy()
            if not np.isfinite(values).all():raise ValueError('Nonfinite MAP block')
            flag=np.frombuffer(raw,dtype=endian+'i4',count=n,offset=size).astype('int32') if grid and len(raw)>size else None
            blocks.append(values);flags.append(flag);shapes.append(shape)
        if f.read(1):raise ValueError('Unexpected trailing MAP records')
    return shapes,blocks,flags


def convert(task):
    row,audit,c,signature,root,output,mappings,resume=task
    sample=row['sample'];folder=Path(output);root=Path(root)
    path=folder/'samples'/f'{sample}.npz';meta_path=folder/'metadata'/f'{sample}.json'
    identity={k:audit[k] for k in ['grid_sha256','solution_sha256']}
    if resume and path.is_file() and meta_path.is_file():
        old=json.loads(meta_path.read_text())
        if (old.get('contract_signature')!=signature or old.get('audited_source_hashes')!=identity
                or old.get('split')!=SPLITS[row['split']]):
            raise ValueError(sample+': existing cache identity differs; use a new output directory')
        return old,'reused'
    grid=map_path(audit['grid_path'],root,mappings)
    solution=map_path(audit['solution_path'],root,mappings)
    declared=root/row['solution_path'].replace('\\','/')
    if solution.resolve()!=declared.resolve():raise ValueError(sample+': audit solution path differs from manifest')
    if grid.name.lower()!='ofmeshplt.xyz':raise ValueError('Only audited output ofmeshplt.xyz is accepted')
    # Optional per-case declared channel order, not a dataset-wide re-audit.
    field_manifest=solution.parent.parent/'flow_visualization/flow_field_manifest.json'
    warnings=[]
    if field_manifest.exists():
        visual=json.loads(field_manifest.read_text(encoding='utf-8-sig'))
        order=visual.get('variable_names')
        for key,expected in [('grid_file',audit['grid_sha256']),('solution_file',audit['solution_sha256'])]:
            recorded=visual.get(key,{}).get('sha256')
            if recorded and recorded!=expected:warnings.append(key+': visualization metadata is stale relative to the audited source; statistics/images ignored')
        if order is not None and order!=c['raw_field_order']:raise ValueError(sample+': declared field order differs')
    gs,coordinates,flags=read_map(grid,True);fs,fields,_=read_map(solution,False)
    if gs!=fs:raise ValueError(sample+': output grid/field block shapes differ')
    xyz=np.concatenate(coordinates)*c['coordinate_multiplier']
    targets=np.concatenate([v[:,c['target_raw_indices']] for v in fields])
    if (targets[:,3:5]<=0).any():raise ValueError(sample+': nonpositive static pressure/temperature')
    counts=[len(v) for v in coordinates];offsets=np.r_[0,np.cumsum(counts,dtype='int64')]
    n=len(xyz)
    # Train-only statistics are accumulated from decoded float64 source values.
    mean=targets.mean(0);m2=((targets-mean)**2).sum(0)
    arrays=dict(coordinates=xyz.astype('float32'),targets=targets.astype('float32'),
        output_names=np.array(c['target_names']),output_units=np.array(c['target_units']),
        coordinate_unit=np.array(c['coordinate_unit']),sample_id=np.array(sample),
        original_node_ids=np.arange(n,dtype='int64'),block_shapes=np.asarray(gs,dtype='int32'),
        block_offsets=offsets,block_ids=np.repeat(np.arange(len(gs),dtype='int32'),counts),
        global_conditions=np.array(c['boundary_values'],dtype='float32'),
        global_condition_names=np.array(c['boundary_names']),global_condition_units=np.array(c['boundary_units']),
        grid_integer_flags=np.concatenate([flag if flag is not None else np.zeros(count,dtype='int32') for flag,count in zip(flags,counts)]),
        grid_flags_present=np.concatenate([np.full(count,flag is not None,dtype=bool) for flag,count in zip(flags,counts)]))
    if not np.isfinite(arrays['coordinates']).all() or not np.isfinite(arrays['targets']).all():raise ValueError('Float32 conversion overflow')
    flag_counts={}
    for flag in flags:
        if flag is not None:
            values,amounts=np.unique(flag,return_counts=True)
            for v,amount in zip(values,amounts):flag_counts[str(v)]=flag_counts.get(str(v),0)+int(amount)
    metadata=dict(sample=sample,split=SPLITS[row['split']],original_split=row['split'],grade='A',
        source_grid=str(grid),source_solution=str(solution),audited_source_hashes=identity,
        source_hash_policy='Trusted completed audit, not recalculated during conversion',
        warnings=warnings,contract_signature=signature,nodes=n,block_shapes=gs,block_offsets=offsets.tolist(),
        node_order='block order; within each block: i fastest, then j, then k; zero-based IDs',
        integer_flags=flag_counts,integer_flag_policy=c['integer_flag_policy'],
        duplicate_coordinate_policy=c['duplicate_coordinate_policy'],label_quality=c['label_quality'],
        target_statistics=dict(count=n,mean=mean.tolist(),m2=m2.tolist(),minimum=targets.min(0).tolist(),maximum=targets.max(0).tolist()),
        coordinate_bounds=dict(minimum=xyz.min(0).tolist(),maximum=xyz.max(0).tolist()),
        float32_max_abs_rounding=dict(coordinates=float(np.abs(xyz-arrays['coordinates']).max()),
                                     targets=np.abs(targets-arrays['targets']).max(0).tolist()),
        cache_file=str(path.resolve()))
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as f:
        temporary=Path(f.name);np.savez(f,**arrays)
    temporary.replace(path);atomic_json(meta_path,metadata)
    return metadata,'converted'


def combine_training(records):
    count=0;mean=np.zeros(5);m2=np.zeros(5);lo=np.full(3,np.inf);hi=np.full(3,-np.inf)
    for r in sorted(records,key=lambda r:r['sample']):
        if r['split']!='train':continue
        s=r['target_statistics'];n=s['count'];delta=np.asarray(s['mean'])-mean
        m2+=np.asarray(s['m2'])+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n);count+=n
        lo=np.minimum(lo,r['coordinate_bounds']['minimum']);hi=np.maximum(hi,r['coordinate_bounds']['maximum'])
    if count==0:return None
    return dict(source_split='train',nodes=count,mean=mean.tolist(),std=np.sqrt(m2/count).tolist(),
                coordinate_minimum=lo.tolist(),coordinate_maximum=hi.tolist(),
                note='Statistics only; physical cached values are not normalized. No validation/test/baseline data included.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--contract',type=Path,default=ROOT/'configs/datasets/cr15_map_v1.json')
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--samples',nargs='+',help='Explicit partial smoke-test IDs; partial outputs are not complete caches')
    parser.add_argument('--path-map',action='append',default=[],metavar='OLD=NEW')
    args=parser.parse_args()
    if args.workers<1:parser.error('workers must be positive')
    if not args.plan_only and args.audit is None:parser.error('Conversion requires the completed --audit report')
    mappings=[]
    for value in args.path_map:
        if '=' not in value:parser.error('path-map must be OLD=NEW')
        mappings.append(value.split('=',1))
    root=args.root.expanduser().resolve();output=args.output.expanduser().resolve()
    c=load_contract(args.contract);signature=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    rows,eligible,excluded,audited=selection(root,args.audit.expanduser().resolve() if args.audit else None)
    if output.exists() and not args.resume:parser.error('Output exists; use --resume or choose a new directory')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=c:raise ValueError('Existing output has different contract')
    atomic_json(output/'contract.json',c)
    selection_keys=[k for k in ['sample','stream','family','split','level','beta2e_deg','mesh_config',
        'mesh_gate_ok','max_skewness','min_ortho_deg','max_aspect'] if k in rows[0]]
    write_csv(output/'eligible_A.csv',[{k:r[k] for k in selection_keys} for r in eligible],fields=selection_keys)
    write_csv(output/'excluded_B.csv',[{k:r[k] for k in selection_keys} for r in excluded],fields=selection_keys)
    write_csv(output/'diagnostics_only.csv',rows)
    split_counts={split:sum(SPLITS[r['split']]==split for r in eligible) for split in SPLITS.values()}
    plan=dict(schema=c['schema'],dataset_root=str(root),manifest=str(root/'dataset_manifest.csv'),
        audit_directory=str(args.audit) if args.audit else None,total=len(rows),eligible_A=len(eligible),excluded_B=len(excluded),
        splits=split_counts,contract_signature=signature,
        inputs=['coordinates','global_conditions'],targets=c['target_names'],
        diagnostic_columns_are_not_inputs=True)
    atomic_json(output/'plan.json',plan)
    print(json.dumps(plan,indent=2),flush=True)
    if args.plan_only:
        atomic_json(output/'summary.json',dict(state='plan_only',cache_ready=False,**plan));return
    wanted=set(args.samples or [r['sample'] for r in eligible])
    if not wanted.issubset({r['sample'] for r in eligible}):raise ValueError('Requested sample not A-grade or missing')
    (output/'samples').mkdir(exist_ok=True);(output/'metadata').mkdir(exist_ok=True)
    tasks=[(r,audited[r['sample']],c,signature,root,output,mappings,args.resume) for r in eligible if r['sample'] in wanted]
    started=time.perf_counter();records=[]
    atomic_json(output/'summary.json',dict(state='converting',cache_ready=False,**plan))
    try:
        if args.workers==1:
            iterator=map(convert,tasks)
            for metadata,action in iterator:
                records.append(metadata);print(f'[{len(records)}/{len(tasks)}] {metadata["sample"]}: {action}, nodes={metadata["nodes"]}',flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures=[pool.submit(convert,task) for task in tasks]
                for future in as_completed(futures):
                    metadata,action=future.result();records.append(metadata)
                    print(f'[{len(records)}/{len(tasks)}] {metadata["sample"]}: {action}, nodes={metadata["nodes"]}',flush=True)
        complete=len(records)==len(eligible)
        splits={key:[f'samples/{r["sample"]}.npz' for r in eligible if r['sample'] in wanted and SPLITS[r['split']]==key] for key in SPLITS.values()}
        atomic_json(output/'splits.json',splits)
        train_stats=combine_training(records)
        if train_stats:atomic_json(output/'train_statistics.json',train_stats)
        summary=dict(state='complete' if complete else 'partial',cache_ready=complete,
            converted_or_reused=len(records),elapsed_seconds=time.perf_counter()-started,**plan)
        atomic_json(output/'summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
    except Exception as error:
        atomic_json(output/'summary.json',dict(state='failed',cache_ready=False,error=str(error),**plan));raise


if __name__=='__main__':main()
