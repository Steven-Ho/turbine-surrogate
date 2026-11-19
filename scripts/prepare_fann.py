"""Extract a common HEXA_8 subset from one faNN shard row, retaining node IDs.
Use identical --cells across all cases; this is a subdomain protocol, not full-field training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scripts.fann_format import ArrayUnpickler, FIELDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/fann_small'))
    p.add_argument('--parquet',required=True,help='Shard filename relative to source, e.g. data/hf_train-....parquet')
    p.add_argument('--row',type=int,default=0)
    p.add_argument('--cells',type=int,default=512)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.cells<=10000:p.error('Local subdomain protocol requires 1..10000 cells')
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
    ids,inv=np.unique(conn[:a.cells].reshape(-1)-1,return_inverse=True)
    coordinates=np.column_stack([read(prefix+'GridCoordinates/Coordinate'+axis)[ids] for axis in 'XYZ'])
    targets=np.column_stack([read(prefix+'VertexFields/'+field)[ids] for field in FIELDS])
    bc=np.array([read('Global/RotatingVelocityX')[0],read('Global/OutletPressure')[0]],dtype=np.float32)
    assert np.isfinite(coordinates).all() and np.isfinite(targets).all() and np.isfinite(bc).all()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f:
        np.savez_compressed(f,coordinates=coordinates,targets=targets,hexahedra=inv.reshape(-1,8),
            original_node_ids=ids,original_cell_ids=np.arange(a.cells),output_names=np.array(FIELDS),
            operating_conditions=bc,geometry_id=read('Global/GeometryNumber'),
            source_case=np.array(f'{a.parquet}:row{a.row}'))
    print(json.dumps({'file':str(a.output),'nodes':len(ids),'cells':a.cells,'bytes':a.output.stat().st_size}))

if __name__=='__main__':main()
