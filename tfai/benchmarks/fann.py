"""Six-field faNN adapters on a common, explicitly selected volume submesh.

POD fits a basis on training snapshots only. UNet uses a Cartesian embedding
with nearest-node geometry features and trilinear decoding at the original nodes.
All methods return the same ordered nodal output; targets never enter forward.
"""
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data

MODELS = ('POD', 'UNet', 'GCN', 'Graph_UNet', 'Graph_Transformer', 'Transolver_3', 'GeoTransolver')
FIELDS = ('VelocityX', 'VelocityY', 'VelocityZ', 'Pressure', 'Temperature', 'Density')
PAIRS = np.array([(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)])


def load_sample(path):
    with np.load(path, allow_pickle=False) as f:
        s = {k:f[k].copy() for k in f.files}
    if tuple(s['output_names']) != FIELDS: raise ValueError('Unexpected output order')
    n = len(s['coordinates'])
    if s['coordinates'].shape != (n,3) or s['targets'].shape != (n,6): raise ValueError('Invalid fields')
    if s['operating_conditions'].shape != (2,): raise ValueError('Expected rotation rad/s and outlet pressure Pa')
    cells = s['hexahedra']
    if cells.ndim != 2 or cells.shape[1] != 8 or cells.dtype.kind not in 'iu': raise ValueError('Invalid HEXA_8')
    if cells.min() < 0 or cells.max() >= n: raise ValueError('Invalid connectivity')
    if len(np.unique(s['original_node_ids'])) != n: raise ValueError('Duplicate node IDs')
    for k in ('coordinates','targets','operating_conditions'):
        if not np.isfinite(s[k]).all(): raise ValueError('Nonfinite '+k)
    h=hashlib.sha256()
    for key in ('coordinates','targets','operating_conditions'):
        h.update(np.ascontiguousarray(s[key]).tobytes())
    s['fingerprint']=h.hexdigest()
    s['source'] = str(Path(path).resolve())
    return s


class Normalizer:
    def __init__(self, train):
        # Training snapshots only. Fixed physical scales prevent zero-variance
        # single-snapshot channels from causing huge normalized values.
        xyz = np.concatenate([s['coordinates'] for s in train])
        self.origin = xyz.min(0)
        self.length = max(float(np.ptp(xyz,axis=0).max()),1e-8)
        values = np.concatenate([s['targets'] for s in train]).astype(np.float64)
        self.mean = values.mean(0).astype(np.float32)
        floor = np.array([1,1,1,100,1,.01])
        self.scale = np.maximum(values.std(0),floor).astype(np.float32)

    def graph(self,s,device):
        xyz = ((s['coordinates']-self.origin)/self.length).astype(np.float32)
        bc = (s['operating_conditions']/np.array([2000,1e5])).astype(np.float32)
        x = np.column_stack([xyz,np.broadcast_to(bc,(len(xyz),2))]).astype(np.float32)
        edges = s['hexahedra'][:,PAIRS].reshape(-1,2)
        edges = np.unique(np.concatenate([edges,edges[:,::-1]]),axis=0)
        delta = xyz[edges[:,1]]-xyz[edges[:,0]]
        return Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges.T.copy()).long(),
                    edge_attr=torch.from_numpy(np.column_stack([delta,np.linalg.norm(delta,axis=1)]))).to(device)

    def target(self,s,device):
        return torch.as_tensor((s['targets']-self.mean)/self.scale,device=device)

    def decode(self,pred):
        return pred.detach().cpu().numpy()*self.scale+self.mean


class FannAdapter(nn.Module):
    def __init__(self,name,train,norm,scale='small',grid=32,rank=32):
        super().__init__()
        if name not in MODELS: raise ValueError(name)
        self.name=name;self.grid=grid
        cfg={'model':{'scale':scale,'geotransolver_backend':'portable_torch'},'training':{}}
        if name == 'POD':
            from tfai.models.pod import POD
            ref=train[0]
            for s in train:
                if not np.array_equal(s['original_node_ids'],ref['original_node_ids']) or not np.array_equal(s['hexahedra'],ref['hexahedra']):
                    raise ValueError('POD requires aligned node IDs and identical submesh topology')
            snapshots=torch.stack([norm.target(s,'cpu').flatten() for s in train])
            # Uncentered POD after per-channel training normalization supports a
            # single snapshot for smoke only; useful ranks need multiple cases.
            _,_,vh=torch.linalg.svd(snapshots,full_matrices=False)
            basis=vh[:min(rank,len(vh))].T.contiguous()
            self.register_buffer('basis',basis)
            self.register_buffer('node_ids',torch.from_numpy(ref['original_node_ids'].astype(np.int64)))
            self.net=POD(3*len(ref['coordinates'])+2,basis.shape[1],scale,cfg)
        elif name=='UNet':
            from tfai.models.unet import UNet
            if grid < 32 or grid%16: raise ValueError('UNet grid must be >=32 and divisible by 16')
            self.net=UNet(6,6,(grid,grid,grid),scale)
        else:
            if name=='GCN':
                from tfai.models.gcn import GCN as cls
            elif name=='Graph_UNet':
                from tfai.models.graph_unet import GraphUNet as cls
            elif name=='Graph_Transformer':
                from tfai.models.graph_transformer import GraphTransformer as cls
            else:
                from tfai.models.transolver_adapters import Transolver3,GeoTransolver
                cls=Transolver3 if name=='Transolver_3' else GeoTransolver
            self.net=cls(config=cfg,input_dim=5,output_dim=6)

    def prepare(self,graph,node_ids):
        if self.name=='POD':
            if not torch.equal(torch.as_tensor(node_ids,device=self.node_ids.device),self.node_ids):
                raise ValueError('POD node ordering differs from fitted basis')
            return torch.cat([graph.x[:,:3].flatten(),graph.x[0,3:]]).unsqueeze(0)
        if self.name!='UNet': return graph
        xyz=graph.x[:,:3];lo=xyz.min(0).values;span=(xyz.max(0).values-lo).clamp_min(1e-8)
        axes=[torch.linspace(lo[i],lo[i]+span[i],self.grid,device=xyz.device) for i in range(3)]
        # Tensor axes D/H/W correspond to z/y/x; grid_sample expects x/y/z.
        zz,yy,xx=torch.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
        positions=torch.stack([xx,yy,zz],-1)
        distance,ids=cKDTree(xyz.cpu().numpy()).query(positions.cpu().numpy().reshape(-1,3))
        nearest=graph.x[torch.as_tensor(ids,device=xyz.device)]
        features=torch.cat([nearest,torch.as_tensor(distance,device=xyz.device,dtype=xyz.dtype)[:,None]],1)
        volume=features.T.reshape(1,6,self.grid,self.grid,self.grid)
        query=((xyz-lo)/span*2-1).reshape(1,1,1,-1,3)
        return volume,query

    def forward(self,inputs):
        if self.name=='POD': return (self.net(inputs)@self.basis.T).reshape(-1,6)
        if self.name=='UNet':
            volume,query=inputs
            return F.grid_sample(self.net(volume),query,mode='bilinear',align_corners=True).reshape(6,-1).T
        return self.net(inputs)
