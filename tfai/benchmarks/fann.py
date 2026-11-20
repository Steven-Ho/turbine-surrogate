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


from functools import lru_cache

@lru_cache(maxsize=2)
def _load_shared_mesh(path,stamp=None):
    with np.load(path,allow_pickle=False) as data:
        result={key:data[key] for key in ('hexahedra','original_node_ids','original_cell_ids')}
    for value in result.values():value.flags.writeable=False
    return result


def load_sample(path, validate=True):
    with np.load(path, allow_pickle=False) as f:
        s = {k:f[k].copy() for k in f.files}
    if 'mesh_file' in s:
        mesh=Path(str(s['mesh_file']))
        if not mesh.is_absolute():mesh=Path(path).parent/mesh
        stamp=mesh.stat()
        s.update(_load_shared_mesh(str(mesh.resolve()),(stamp.st_dev,stamp.st_ino,stamp.st_size,stamp.st_mtime_ns,stamp.st_ctime_ns)))
    if not validate:
        s['source']=str(Path(path).resolve())
        return s
    if tuple(s['output_names']) != FIELDS: raise ValueError('Unexpected output order')
    n = len(s['coordinates'])
    if s['coordinates'].shape != (n,3) or s['targets'].shape != (n,6): raise ValueError('Invalid fields')
    if s['operating_conditions'].shape != (2,): raise ValueError('Expected rotation rad/s and outlet pressure Pa')
    cells = s['hexahedra']
    if cells.ndim != 2 or cells.shape[1] != 8 or cells.dtype.kind not in 'iu': raise ValueError('Invalid HEXA_8')
    if cells.min() < 0 or cells.max() >= n: raise ValueError('Invalid connectivity')
    node_ids=s['original_node_ids']
    if node_ids.shape!=(n,) or node_ids.dtype.kind not in 'iu' or (node_ids<0).any() or len(np.unique(node_ids))!=n:
        raise ValueError('Invalid or duplicate node IDs')
    if str(s.get('selection_protocol',''))=='full':
        if int(s.get('full_domain_nodes',-1))!=n or not np.array_equal(node_ids,np.arange(n)):
            raise ValueError('Full-domain sample does not contain all original nodes')
    for k in ('coordinates','targets','operating_conditions'):
        if not np.isfinite(s[k]).all(): raise ValueError('Nonfinite '+k)
    h=hashlib.sha256()
    for key in ('coordinates','targets','operating_conditions','original_node_ids','hexahedra','wall_distance','fixed_conditions'):
        if key not in s: continue
        h.update((key+str(s[key].shape)+str(s[key].dtype)).encode())
        h.update(memoryview(np.ascontiguousarray(s[key])).cast('B'))
    if 'wall_distance' in s:
        if s['wall_distance'].shape != (n,) or not np.isfinite(s['wall_distance']).all() or (s['wall_distance'] < 0).any():
            raise ValueError('Invalid wall distance')
    s['fingerprint']=h.hexdigest()
    s['source'] = str(Path(path).resolve())
    return s


def normalization_summary(sample):
    values=sample['targets'].astype(np.float64)
    mean=values.mean(0)
    return {'lo':sample['coordinates'].min(0),'hi':sample['coordinates'].max(0),
            'count':len(values),'mean':mean,'m2':((values-mean)**2).sum(0),
            'wall_distance':'wall_distance' in sample}


class NormalizationStats:
    """Merge per-case training summaries in the original deterministic order."""
    def __init__(self):
        self.lo=np.full(3,np.inf);self.hi=np.full(3,-np.inf)
        self.count=0;self.mean=np.zeros(6);self.m2=np.zeros(6);self.wall=None

    def add(self,summary):
        wall=summary['wall_distance']
        if self.wall is None:self.wall=wall
        if wall!=self.wall:raise ValueError('Wall-distance schema must match across training samples')
        self.lo=np.minimum(self.lo,summary['lo']);self.hi=np.maximum(self.hi,summary['hi'])
        n=summary['count'];delta=summary['mean']-self.mean
        self.m2+=summary['m2']+delta**2*self.count*n/(self.count+n)
        self.mean+=delta*n/(self.count+n);self.count+=n

    def state(self):
        if not self.count:raise ValueError('No training data for normalization')
        return {'origin':self.lo.astype(np.float32),'length':max(float((self.hi-self.lo).max()),1e-8),
                'mean':self.mean.astype(np.float32),
                'scale':np.maximum(np.sqrt(self.m2/self.count),[1,1,1,100,1,.01]).astype(np.float32),
                'wall_distance':self.wall}


class Normalizer:
    def __init__(self, train, progress=None):
        # Training snapshots only, with identical ordered float64 reductions.
        stats=NormalizationStats()
        for sample_index,sample in enumerate(train):
            if progress is not None and sample_index%10==0:
                progress(f'normalization sample {sample_index+1}/{len(train)}')
            stats.add(normalization_summary(sample))
        self.__dict__.update(stats.state())

    @classmethod
    def from_state(cls, state):
        obj = cls.__new__(cls)
        obj.origin = np.asarray(state['origin'], dtype=np.float32)
        obj.mean = np.asarray(state['mean'], dtype=np.float32)
        obj.scale = np.asarray(state['scale'], dtype=np.float32)
        obj.length = float(state['length']); obj.wall_distance = bool(state.get('wall_distance',False))
        if obj.origin.shape != (3,) or obj.mean.shape != (6,) or obj.scale.shape != (6,) or obj.length <= 0:
            raise ValueError('Invalid saved normalization')
        if not np.isfinite(np.r_[obj.origin,obj.mean,obj.scale,obj.length]).all() or (obj.scale<=0).any():
            raise ValueError('Nonfinite saved normalization')
        return obj

    def graph(self,s,device,with_edges=True):
        xyz = ((s['coordinates']-self.origin)/self.length).astype(np.float32)
        bc = (s['operating_conditions']/np.array([2000,1e5])).astype(np.float32)
        features = [xyz]
        if self.wall_distance:
            if 'wall_distance' not in s: raise ValueError('Missing wall distance')
            features.append((s['wall_distance']/self.length)[:,None])
        x = np.column_stack(features+[np.broadcast_to(bc,(len(xyz),2))]).astype(np.float32)
        if not with_edges:return Data(x=torch.from_numpy(x)).to(device)
        edges = s['hexahedra'][:,PAIRS].reshape(-1,2)
        edges = np.unique(np.concatenate([edges,edges[:,::-1]]),axis=0)
        delta = xyz[edges[:,1]]-xyz[edges[:,0]]
        return Data(x=torch.from_numpy(x),edge_index=torch.from_numpy(edges.T.copy()).long(),
                    edge_attr=torch.from_numpy(np.column_stack([delta,np.linalg.norm(delta,axis=1)]))).to(device)

    def target(self,s,device):
        return torch.as_tensor((s['targets']-self.mean)/self.scale,device=device)

    def decode(self,pred):
        return pred.detach().cpu().numpy()*self.scale+self.mean


def voxel_queries(xyz,grid):
    lo=xyz.min(0).values;span=(xyz.max(0).values-lo).clamp_min(1e-8)
    axes=[torch.linspace(lo[i],lo[i]+span[i],grid,device=xyz.device) for i in range(3)]
    zz,yy,xx=torch.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
    return torch.stack([xx,yy,zz],-1),lo,span


class FannAdapter(nn.Module):
    def __init__(self,name,train,norm,scale='small',grid=32,rank=32,pod_state=None,pod_geometry_bins=0):
        super().__init__()
        if name not in MODELS: raise ValueError(name)
        self.name=name;self.grid=grid
        self.input_dim=6 if norm.wall_distance else 5
        self.pod_geometry_bins=int(pod_geometry_bins)
        cfg={'model':{'scale':scale,'geotransolver_backend':'portable_torch'},'training':{}}
        if name == 'POD':
            from tfai.models.pod import POD
            if pod_state is None:
                ref=train[0]
                for sample in train:
                    if not np.array_equal(sample['original_node_ids'],ref['original_node_ids']) or not np.array_equal(sample['hexahedra'],ref['hexahedra']):
                        raise ValueError('POD requires aligned node IDs and identical submesh topology')
                snapshots=torch.stack([norm.target(sample,'cpu').flatten() for sample in train])
                _,_,vh=torch.linalg.svd(snapshots,full_matrices=False)
                basis=vh[:min(rank,len(vh))].T.contiguous()
                node_ids=torch.from_numpy(ref['original_node_ids'].astype(np.int64))
            else:
                basis=pod_state['basis'].detach().cpu()
                node_ids=pod_state['node_ids'].detach().cpu()
                if basis.ndim!=2 or basis.shape[0]!=len(node_ids)*6:
                    raise ValueError('Invalid saved POD basis')
            self.register_buffer('basis',basis)
            self.register_buffer('node_ids',node_ids)
            self.net=POD((self.input_dim-2)*(self.pod_geometry_bins or len(node_ids))+2,basis.shape[1],scale,cfg)
        elif name=='UNet':
            from tfai.models.unet import UNet
            if grid < 32 or grid%16: raise ValueError('UNet grid must be >=32 and divisible by 16')
            self.net=UNet(self.input_dim+1,6,(grid,grid,grid),scale)
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
            self.net=cls(config=cfg,input_dim=self.input_dim,output_dim=6)

    def prepare(self,graph,node_ids):
        if self.name=='POD':
            if not torch.equal(torch.as_tensor(node_ids,device=self.node_ids.device),self.node_ids):
                raise ValueError('POD node ordering differs from fitted basis')
            geometry=graph.x[:,:-2]
            if self.pod_geometry_bins:
                geometry=F.adaptive_avg_pool1d(geometry.T[None],self.pod_geometry_bins).squeeze(0).T
            return torch.cat([geometry.flatten(),graph.x[0,-2:]]).unsqueeze(0)
        if self.name!='UNet': return graph
        xyz=graph.x[:,:3];positions,lo,span=voxel_queries(xyz,self.grid)
        cache=getattr(self,'unet_mapping_cache',None)
        if cache:
            from .fann_unet_cache import nearest_mapping
            distance,ids=nearest_mapping(xyz.cpu().numpy(),positions.cpu().numpy().reshape(-1,3),cache)
        else:
            distance,ids=cKDTree(xyz.cpu().numpy()).query(positions.cpu().numpy().reshape(-1,3))
        nearest=graph.x[torch.as_tensor(ids,device=xyz.device)]
        features=torch.cat([nearest,torch.as_tensor(distance,device=xyz.device,dtype=xyz.dtype)[:,None]],1)
        volume=features.T.reshape(1,self.input_dim+1,self.grid,self.grid,self.grid)
        query=((xyz-lo)/span*2-1).reshape(1,1,1,-1,3)
        return volume,query

    def forward(self,inputs):
        if self.name=='POD': return (self.net(inputs)@self.basis.T).reshape(-1,6)
        if self.name=='UNet':
            volume,query=inputs
            return F.grid_sample(self.net(volume),query,mode='bilinear',align_corners=True).reshape(6,-1).T
        return self.net(inputs)
