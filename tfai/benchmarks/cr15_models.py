"""CR15 five-field adapters for the six existing non-GeoTransolver baselines.

All use actual case geometry. POD uses a training-only reference-mesh projection;
UNet uses a Cartesian embedding. Graph models retain full-case nodal outputs.
"""
from contextlib import nullcontext
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import Data
from .cr15 import CR15Normalizer, CR15Store, CR15GeoTransolver
from .fann import voxel_queries
from .fann_graph_execution import BoundedGraphExecution, atomic_npz, csr_from_hexahedra
from .fann_unet_cache import nearest_mapping
from .fann_logging import print

MODELS=('POD','UNet','GCN','Graph_UNet','Graph_Transformer','Transolver_3','GeoTransolver')
GRAPH_MODELS={'GCN','Graph_UNet','Graph_Transformer'}


def hexahedra(shapes,offsets,nodes):
    shapes=np.asarray(shapes,dtype=np.int64);offsets=np.asarray(offsets,dtype=np.int64)
    if shapes.ndim!=2 or shapes.shape[1]!=3 or (shapes<2).any():raise ValueError('Expected 3D structured CR15 blocks')
    if not np.array_equal(offsets,np.r_[0,np.cumsum(np.prod(shapes,axis=1))]) or offsets[-1]!=nodes:raise ValueError('Block offsets do not cover all nodes')
    blocks=[]
    for (ni,nj,nk),offset in zip(shapes,offsets):
        ids=np.arange(ni*nj*nk).reshape(nk,nj,ni)+offset # MAP i varies fastest
        blocks.append(np.stack([ids[:-1,:-1,:-1],ids[:-1,:-1,1:],ids[:-1,1:,1:],ids[:-1,1:,:-1],
                                ids[1:,:-1,:-1],ids[1:,:-1,1:],ids[1:,1:,1:],ids[1:,1:,:-1]],axis=-1).reshape(-1,8))
    return np.concatenate(blocks)


class CR15GraphExecution(BoundedGraphExecution):
    def adjacency(self,sample):
        if self.cells is sample['coordinates'] and self.topology is not None:return self.topology
        digest=hashlib.sha256(b'cr15-structured-exact-seams-v1')
        for key in ['coordinates','block_shapes','block_offsets']:
            arr=np.ascontiguousarray(sample[key]);digest.update(str((arr.shape,arr.dtype.str)).encode());digest.update(arr.tobytes())
        token=digest.hexdigest();path=self.cache_dir/(token+'.npz')
        if token!=self.key:
            with (self.cache_dir/(token+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if path.exists():adj=sparse.load_npz(path)
                else:
                    n=len(sample['coordinates']);cells=hexahedra(sample['block_shapes'],sample['block_offsets'],n)
                    adj=csr_from_hexahedra(cells,n)
                    # Chain coincident vertices at block interfaces, without merging output IDs.
                    _,inverse,counts=np.unique(sample['coordinates'],axis=0,return_inverse=True,return_counts=True)
                    order=np.argsort(inverse,kind='stable');left=order[:-1];right=order[1:];same=inverse[left]==inverse[right]
                    left=left[same];right=right[same]
                    if len(left):
                        seam=sparse.coo_matrix((np.ones(2*len(left),dtype=np.float32),(np.r_[left,right],np.r_[right,left])),shape=(n,n)).tocsr()
                        adj=(adj+seam).tocsr();adj.data.fill(1.);adj.sort_indices()
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir,suffix='.npz',delete=False) as stream:temporary=Path(stream.name)
                    sparse.save_npz(temporary,adj,compressed=False);temporary.replace(path)
                if adj.shape!=(len(sample['coordinates']),)*2:raise ValueError('Invalid CR15 graph cache')
            self.topology=adj;self.key=token;self.parts.clear();self.part_bytes=0
            self.base_normalized=None;self.base_augmented=None
            print(f'{self.name}: CR15 graph nodes={adj.shape[0]} directed_edges={adj.nnz} cache={path}',flush=True)
        self.cells=sample['coordinates'];return self.topology


class GraphNormalizer:
    def __init__(self,norm):self.norm=norm
    def graph(self,sample,device,with_edges=False):
        if with_edges:raise ValueError('Use bounded graph execution')
        return self.norm.input(sample,device)


def fit_pod(prepared,norm,rank,cache):
    """Streaming snapshot POD on the first training mesh; never read valid/test targets."""
    folder=Path(cache)/'pod';folder.mkdir(parents=True,exist_ok=True)
    identity=hashlib.sha256((prepared['identity']+f'cr15-reference-pod-v1-{rank}').encode()).hexdigest()
    path=folder/(identity+'.npz')
    with (folder/(identity+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            with np.load(path) as data:return {key:data[key] for key in data.files}
        store=CR15Store(prepared['paths']['train']);ref=store[0];xyz=(ref['coordinates']-norm.origin)/norm.length
        dofs=len(xyz)*5;cases=len(store)
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            snapshots=np.memmap(Path(scratch)/'snapshots.f32',mode='w+',dtype=np.float32,shape=(cases,dofs))
            for i,sample in enumerate(store):
                coords=(sample['coordinates']-norm.origin)/norm.length
                _,ids=nearest_mapping(coords,xyz,Path(cache)/'pod_mapping')
                snapshots[i]=((sample['targets'][ids]-norm.mean)/norm.scale).reshape(-1)
                if i%20==0:print(f'CR15 POD training snapshots {i+1}/{cases}',flush=True)
            mean=np.empty(dofs,dtype=np.float32);gram=np.zeros((cases,cases),dtype=np.float64)
            for start in range(0,dofs,327680):
                block=np.asarray(snapshots[:,start:start+327680],dtype=np.float64)
                mu=block.mean(0);mean[start:start+len(mu)]=mu;block-=mu;gram+=block@block.T
            values,vectors=np.linalg.eigh(gram);order=np.argsort(values)[::-1]
            keep=order[values[order]>max(float(values.max()),1.)*1e-12][:rank]
            if not len(keep):raise ValueError('POD training fields have zero rank')
            transform=vectors[:,keep]/np.sqrt(values[keep])[None,:]
            basis=np.empty((dofs,len(keep)),dtype=np.float32)
            for start in range(0,dofs,327680):
                block=np.asarray(snapshots[:,start:start+327680],dtype=np.float64)-mean[start:start+327680]
                basis[start:start+len(block.T)]=block.T@transform
            del snapshots
        state=dict(reference=xyz.astype(np.float32),basis=basis,offset=mean)
        atomic_npz(path,**state);print(f'CR15 POD shared training-only basis: {path}',flush=True)
        return state


class CR15Baseline(nn.Module):
    def __init__(self,name,prepared,config,pod_state=None):
        super().__init__();self.name=name;self.amp=config['amp'];self.config=config;self.norm=CR15Normalizer(prepared)
        self.cache=Path(config['model_cache']);self.cache.mkdir(parents=True,exist_ok=True)
        scale=config['scale'];cfg={'model':{'scale':scale},'training':{}}
        self.graph_norm=GraphNormalizer(self.norm)
        if name=='POD':
            state=pod_state if pod_state is not None else fit_pod(prepared,self.norm,config['pod_rank'],self.cache)
            for key in ['reference','basis','offset']:self.register_buffer(key,torch.as_tensor(state[key]))
            from tfai.models.pod import POD
            self.net=POD(config['geometry_bins']*3+6,self.basis.shape[1],scale,cfg)
        elif name=='UNet':
            from tfai.models.unet import UNet
            grid=config['grid'];self.net=UNet(10,5,(grid,)*3,scale)
        elif name=='Transolver_3':
            from tfai.models.transolver_adapters import Transolver3
            cfg['model']['transolver_chunk_size']=config['chunk'];self.net=Transolver3(cfg,9,5)
        else:
            if name=='GCN':from tfai.models.gcn import GCN as cls
            elif name=='Graph_UNet':from tfai.models.graph_unet import GraphUNet as cls
            elif name=='Graph_Transformer':from tfai.models.graph_transformer import GraphTransformer as cls
            else:raise ValueError(name)
            self.net=cls(cfg,9,5)
            self.executor=CR15GraphExecution(name,{'graph_owner_nodes':config['graph_owner_nodes'],'graph_edge_chunk':config['graph_edge_chunk'],'graph_cache_mb':512},self.cache/'graphs')

    def forward(self,data):
        context=torch.autocast(device_type=data.x.device.type,dtype=torch.bfloat16) if self.amp=='bf16' else nullcontext()
        with context:result=self._forward(data)
        return result.float()

    def _forward(self,data):
        x=data.x;device=x.device
        if self.name=='POD':
            xyz=x[:,:3].detach().cpu().numpy();ref=self.reference.detach().cpu().numpy()
            _,to_ref=nearest_mapping(xyz,ref,self.cache/'pod_mapping')
            _,to_case=nearest_mapping(ref,xyz,self.cache/'pod_mapping')
            geometry=x[torch.as_tensor(to_ref,device=device),:3]
            geometry=F.adaptive_avg_pool1d(geometry.T[None],self.config['geometry_bins']).flatten()
            features=torch.cat([geometry,x[0,-6:]])[None]
            reference=(self.net(features).float()@self.basis.T+self.offset).reshape(-1,5)
            return reference[torch.as_tensor(to_case,device=device)]
        if self.name=='UNet':
            xyz=x[:,:3];grid=self.config['grid'];positions,lo,span=voxel_queries(xyz,grid)
            distance,ids=nearest_mapping(xyz.detach().cpu().numpy(),positions.cpu().numpy().reshape(-1,3),self.cache/'unet_mapping')
            features=torch.cat([x[torch.as_tensor(ids,device=device)],torch.as_tensor(distance,device=device,dtype=x.dtype)[:,None]],1)
            volume=features.T.reshape(1,10,grid,grid,grid)
            query=((xyz-lo)/span*2-1).reshape(1,1,1,-1,3)
            field=self.net(volume).float()
            with torch.autocast(device_type=device.type,enabled=False):return F.grid_sample(field,query.float(),align_corners=True).reshape(5,-1).T
        if self.name=='Transolver_3':return self.net(data)
        sample=data.cr15_sample
        if self.name=='Graph_UNet':return self.executor.graph_unet(self,self.graph_norm,sample,device)
        values=[]
        for graph,owners,_ in self.executor.pieces(self,self.graph_norm,sample,device):
            def run(features,edges,attrs,owners):return self.net(Data(x=features,edge_index=edges,edge_attr=attrs))[owners]
            if self.training and torch.is_grad_enabled():value=checkpoint(run,graph.x,graph.edge_index,graph.edge_attr,owners,use_reentrant=False)
            else:value=run(graph.x,graph.edge_index,graph.edge_attr,owners)
            values.append(value)
        return torch.cat(values)


def make_model(prepared,config,saved=None):
    name=config.get('model','GeoTransolver')
    if name=='GeoTransolver':return CR15GeoTransolver(config['scale'],config['amp'],config['chunk'],config['offload'])
    pod_state=None
    if saved and name=='POD':pod_state={key:saved['state_dict'][key].cpu().numpy() for key in ['reference','basis','offset']}
    return CR15Baseline(name,prepared,config,pod_state)


def model_input(model,norm,sample,device):
    result=norm.input(sample,device)
    if getattr(model,'name',None) in GRAPH_MODELS:
        if 'block_shapes' not in sample:
            with np.load(sample['source'],allow_pickle=False) as arrays:
                sample['block_shapes']=arrays['block_shapes'];sample['block_offsets']=arrays['block_offsets']
        result.cr15_sample=sample
    return result
