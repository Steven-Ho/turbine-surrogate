"""Extract a common HEXA_8 subset from one faNN shard row, retaining node IDs.
Use identical --cells across all cases; this is a subdomain protocol, not full-field training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scripts.fann_format import ArrayUnpickler, FIELDS, select_structured_patches


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/fann_small'))
    p.add_argument('--parquet',required=True,help='Shard filename relative to source, e.g. data/hf_train-....parquet')
    p.add_argument('--row',type=int,default=0)
    p.add_argument('--cells',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',choices=['first','structured_patches','full'],default='structured_patches')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--selection-file',type=Path,help='Reuse a common cell-ID selection')
    p.add_argument('--shared-mesh',type=Path,help='Store/reuse topology once for full-domain cases')
    a=p.parse_args()
    if a.selection!='full' and not 1<=a.cells<=10000:p.error('Sampled protocol requires 1..10000 cells; use --selection full for the full mesh')
    if a.output.exists():p.error('Output already exists')
    sidecar=a.source/'tree_constant_part.pkl'
    # Verify known local download manifest before restricted deserialization.
    manifest=a.source/'download_manifest.json'
    entry=next((x for x in json.loads(manifest.read_text()) if x['path']==sidecar.name),None) if manifest.exists() else None
    h=hashlib.sha256()
    with sidecar.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    if entry is not None and h.hexdigest()!=entry['sha256']:raise ValueError('Common mesh checksum mismatch')
    with sidecar.open('rb') as f:constant=ArrayUnpickler(f).load()['hf_train']
    prefix='Base_3_3/Zone/'
    conn=np.asarray(constant[prefix+'HEXA_8/ElementConnectivity']).reshape(-1,8)
    shard=pq.ParquetFile(a.source/a.parquet)
    if not 0<=a.row<shard.metadata.num_rows:p.error('Row out of range')
    # Read only the containing row group and one column at a time.
    offset=0
    for group in range(shard.num_row_groups):
        size=shard.metadata.row_group(group).num_rows
        if a.row<offset+size:break
        offset+=size
    def read(name):return shard.read_row_group(group,columns=[name]).column(0)[a.row-offset].values.to_numpy()
    n=len(read(prefix+'GridCoordinates/CoordinateX'))
    assert conn.min()==1 and conn.max()==n
    if a.selection=='full':
        cell_ids=np.arange(len(conn),dtype=np.int64)
        ids=np.arange(n,dtype=np.int64)
        local_cells=(conn-1).astype(np.int32)
    else:
        if a.selection_file and a.selection_file.exists():
            cell_ids=np.load(a.selection_file,allow_pickle=False)
        elif a.selection=='first':cell_ids=np.arange(a.cells,dtype=np.int64)
        else:
            vertex=prefix+'VertexFields/'
            cell_ids=select_structured_patches(conn,[constant[vertex+axis] for axis in 'ijk'],constant[vertex+'block_num'],a.cells,a.seed)
        if len(cell_ids)!=a.cells or len(np.unique(cell_ids))!=len(cell_ids) or cell_ids.min()<0 or cell_ids.max()>=len(conn):
            raise ValueError('Invalid common cell selection')
        if a.selection_file and not a.selection_file.exists():
            a.selection_file.parent.mkdir(parents=True,exist_ok=True)
            np.save(a.selection_file,cell_ids,allow_pickle=False)
        ids,inv=np.unique(conn[cell_ids].reshape(-1)-1,return_inverse=True)
        local_cells=inv.reshape(-1,8)
    coordinates=np.column_stack([read(prefix+'GridCoordinates/Coordinate'+axis)[ids] for axis in 'XYZ'])
    targets=np.column_stack([read(prefix+'VertexFields/'+field)[ids] for field in FIELDS])
    bc=np.array([read('Global/RotatingVelocityX')[0],read('Global/OutletPressure')[0]],dtype=np.float32)
    assert np.isfinite(coordinates).all() and np.isfinite(targets).all() and np.isfinite(bc).all()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(coordinates=coordinates,targets=targets,output_names=np.array(FIELDS),
        wall_distance=read(prefix+'VertexFields/TurbulentDistance')[ids],
        operating_conditions=bc,geometry_id=read('Global/GeometryNumber'),
        source_case=np.array(f'{a.parquet}:row{a.row}'),selection_protocol=np.array(a.selection),
        selection_seed=np.array(a.seed),full_domain_nodes=np.array(n),
        fixed_conditions=np.array([float(np.asarray(constant['Global/'+key]).reshape(-1)[0]) for key in
            ['InletPressureTotal','InletTemperatureTotal','SpecificHeatPressure','SpecificHeatRatio']],dtype=np.float64))
    if a.selection=='full' and a.shared_mesh:
        if not a.shared_mesh.exists():
            a.shared_mesh.parent.mkdir(parents=True,exist_ok=True)
            with a.shared_mesh.open('xb') as f:
                np.savez_compressed(f,hexahedra=local_cells,original_node_ids=ids,original_cell_ids=cell_ids,source_mesh_sha256=np.array(h.hexdigest()))
        with np.load(a.shared_mesh,allow_pickle=False) as mesh:
            if str(mesh['source_mesh_sha256'])!=h.hexdigest():raise ValueError('Shared mesh came from a different source')
        arrays['mesh_file']=np.array(str(a.shared_mesh.resolve()))
    else:arrays.update(hexahedra=local_cells,original_node_ids=ids,original_cell_ids=cell_ids)
    with a.output.open('xb') as f:np.savez_compressed(f,**arrays)
    print(json.dumps({'file':str(a.output),'nodes':len(ids),'cells':len(cell_ids),'bytes':a.output.stat().st_size}))

if __name__=='__main__':main()
