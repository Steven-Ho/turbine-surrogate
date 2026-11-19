"""Validate one faNN Parquet sample and run a six-output GCN smoke on real hex cells.
Requires pyarrow (tested 19.0.1), project dependencies; no network or full-size copies.
"""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
import pyarrow.parquet as pq
from torch_geometric.data import Data
from tfai.models.gcn import GCN

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
                   ('numpy', 'dtype'), ('numpy.core.numeric', '_frombuffer')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')
        return super().find_class(module, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, default=Path('data/fann_small'))
    ap.add_argument('--output', type=Path, default=Path('verification/fann_smoke'))
    ap.add_argument('--cells', type=int, default=512)
    args = ap.parse_args()
    if not 1 <= args.cells <= 10000: ap.error('--cells must be 1..10000')
    args.output.mkdir(parents=True, exist_ok=False)
    # Verify downloaded bytes before reading the vendor pickle.
    for item in json.loads((args.source/'download_manifest.json').read_text()):
        h = hashlib.sha256()
        with (args.source/item['path']).open('rb') as f:
            for b in iter(lambda: f.read(2**20), b''): h.update(b)
        assert h.hexdigest() == item['sha256'], item['path']
    with (args.source/'tree_constant_part.pkl').open('rb') as f:
        constants = ArrayUnpickler(f).load()['hf_train']
    prefix = 'Base_3_3/Zone/'
    conn = np.asarray(constants[prefix+'HEXA_8/ElementConnectivity']).reshape(-1, 8)
    print('Connectivity:', conn.shape, conn.min(), conn.max(), flush=True)
    parquet = pq.ParquetFile(next((args.source/'data').glob('hf_train-*.parquet')))
    assert parquet.metadata.num_rows == 1
    def read(name):
        return parquet.read(columns=[name]).column(0)[0].values.to_numpy()
    n = len(read(prefix+'GridCoordinates/CoordinateX'))
    assert np.issubdtype(conn.dtype, np.integer)
    # CGNS connectivity is one-based; preserve original cell/node IDs in the export.
    assert conn.min() == 1 and conn.max() == n
    node_ids, inverse = np.unique(conn[:args.cells].reshape(-1)-1, return_inverse=True)
    cells = inverse.reshape(-1,8)
    coords = np.column_stack([read(prefix+'GridCoordinates/Coordinate'+a)[node_ids] for a in 'XYZ'])
    fields = ['VelocityX','VelocityY','VelocityZ','Pressure','Temperature','Density']
    targets = np.column_stack([read(prefix+'VertexFields/'+a)[node_ids] for a in fields])
    checks = {}
    for name in parquet.schema_arrow.names:
        a = read(name)
        assert np.isfinite(a).all(), name
        assert len(a) == (n if name.startswith(prefix) else 1), name
        checks[name] = {'length':len(a),'min':float(a.min()),'max':float(a.max())}
    assert (targets[:,3:] > 0).all()
    # Only prescribed operating conditions enter the model; no performance scalars.
    bc = np.array([read('Global/RotatingVelocityX')[0],read('Global/OutletPressure')[0]],dtype=np.float32)
    xyz = ((coords-coords.mean(0))/np.maximum(np.ptp(coords,axis=0),1e-8)).astype(np.float32)
    x = np.column_stack([xyz,np.broadcast_to(bc/np.array([2000,1e5]),(len(xyz),2))]).astype(np.float32)
    pairs = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])
    edges = cells[:,pairs].reshape(-1,2)
    edges = np.unique(np.concatenate([edges,edges[:,::-1]]),axis=0)
    delta = xyz[edges[:,1]]-xyz[edges[:,0]]
    graph = Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges.T.copy()).long(),
                 edge_attr=torch.from_numpy(np.column_stack([delta,np.linalg.norm(delta,axis=1)])))
    torch.set_num_threads(4);torch.manual_seed(42)
    model = GCN({'model':{'scale':'small'}},input_dim=5,output_dim=6)
    y = torch.from_numpy(targets)/torch.tensor([1000,1000,1000,1e5,300,1.0])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
    before = next(model.parameters()).detach().clone();losses=[]
    for _ in range(3):
        optimizer.zero_grad();pred=model(graph)
        assert pred.shape == y.shape and torch.isfinite(pred).all()
        loss = (pred-y).square().mean();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step();losses.append(loss.item())
    assert not torch.equal(before,next(model.parameters()).detach())
    np.savez_compressed(args.output/'submesh.npz',coordinates=coords,hexahedra=cells,
                        original_node_ids=node_ids,original_cell_ids=np.arange(args.cells),targets=targets,
                        output_names=np.array(fields),operating_conditions=bc)
    report = {'status':'passed','purpose':'format and CPU GCN smoke, not accuracy evaluation',
              'source_nodes':n,'source_hex_cells':len(conn),'source_connectivity_base':1,
              'submesh_nodes':len(node_ids),'submesh_cells':args.cells,'directed_edges':len(edges),
              'output_names':fields,'device':'cpu','optimizer_steps':3,'losses':losses,
              'targets_used_as_inputs':False,'field_checks':checks,
              'limitations':['First-cell submesh is not representative of the full domain.',
                             'Only GCN is exercised; seven-model faNN adaptation is not complete.',
                             'Fixed scales are debug scales, not fitted training normalization.']}
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='field_checks'},indent=2))

if __name__ == '__main__': main()
