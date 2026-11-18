"""Verify mesh+BC -> five nodal channels on one complete inlet subdomain.
This is an interface/gradient smoke check, not a trained surrogate evaluation.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch_geometric.data import Data
from tfai.models.gcn import GCN

ROOT=Path(__file__).resolve().parents[1]
PAIRS={10: [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)],
       13: [(0,1),(1,2),(2,0),(3,4),(4,5),(5,3),(0,3),(1,4),(2,5)],
       14: [(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]}
WIDTH={10:4,13:6,14:5}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="auto");args=parser.parse_args()
    manifest=json.loads((ROOT/"data/stator_case_000/manifest.json").read_text(encoding="utf-8"))
    region=manifest["regions"][0];assert region["name"]=="fluid_in" and region["node_offset"]==0 and region["cell_offset"]==0
    n,nc=region["nodes"],region["cells"]
    with h5py.File(manifest["hdf5_path"],"r") as f:
        pos=f["inputs/mesh/positions"][:n].astype(np.float32)
        offsets=f["inputs/mesh/cell_offsets"][:nc+1];types=f["inputs/mesh/cell_types"][:nc]
        conn=f["inputs/mesh/cell_connectivity"][:int(offsets[-1])]
        flags=f["inputs/boundary_flags"][:n]
        globals_=f["inputs/global_bc_values"][...]
        lower=pos.min(0);scale=np.maximum(pos.max(0)-lower,1e-12)
        coordinates=(pos-lower)/scale*2-1
        # Fixed physical unit scales only; no target-dependent values enter x.
        bc_scaled=globals_/np.array([3e6,2000.,3e6,.2,2000.,.2,2000.])
        bits=np.column_stack([(flags & value)!=0 for value in [1,2,4,8,16,32]])
        x=np.column_stack([coordinates,np.broadcast_to(bc_scaled,(n,7)),bits]).astype(np.float32)
        # Keep targets separate from the graph passed to model.forward.
        target=f["targets/uvwpT"][:n].astype(np.float32)
    edges=[]
    for kind in np.unique(types):
        cells=np.flatnonzero(types==kind);nodes=conn[offsets[cells,None]+np.arange(WIDTH[int(kind)])[None,:]]
        edges.append(nodes[:,np.array(PAIRS[int(kind)])].reshape(-1,2))
    edges=np.concatenate(edges);edges=np.unique(np.concatenate([edges,edges[:,::-1]]),axis=0)
    assert edges.min()>=0 and edges.max()<n
    displacement=coordinates[edges[:,1]]-coordinates[edges[:,0]]
    edge_features=np.column_stack([displacement,np.linalg.norm(displacement,axis=1)]).astype(np.float32)
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    torch.manual_seed(42)
    graph=Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges.T.copy()).long(),edge_attr=torch.from_numpy(edge_features)).to(device)
    scales=torch.tensor([1000.,1000.,1000.,3e6,2000.],device=device)
    y=torch.from_numpy(target).to(device)/scales
    model=GCN(config={"model":{"scale":"small"}},input_dim=x.shape[1],output_dim=5).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    optimizer.zero_grad();prediction=model(graph)
    assert tuple(prediction.shape)==(n,5) and torch.isfinite(prediction).all()
    loss=torch.nn.functional.mse_loss(prediction,y);loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    result={"purpose":"interface and finite-gradient smoke only, not generalization or prediction accuracy",
            "source_sample":"stator_case_000","region":"fluid_in","nodes":n,"volume_cells":nc,
            "directed_edges":len(edges),"input_features":x.shape[1],"output_shape":list(prediction.shape),
            "output_names":["u","v","w","p","T"],"device":device,"optimizer_steps":1,
            "finite_loss":bool(torch.isfinite(loss).item()),"scaled_mse_before_step":float(loss.detach().cpu()),
            "targets_used_as_inputs":False,"full_case_nodes":manifest["nodes"]}
    (ROOT/"verification/stator_graph_smoke.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
