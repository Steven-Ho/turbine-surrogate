"""Memory-bounded execution without cropping faNN's nodal objective.

GCN/GraphTransformer: finite-receptive-field owner/halo evaluation and case-level
summed gradients. GraphUNet: original global TopK, CPU adjacency expansion and
chunked constant sparse-linear forward/adjoint. State-dict keys are unchanged.
Floating summation order can differ from the dense PyG implementation.
"""
from collections import OrderedDict
import fcntl
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy import sparse
import torch
from torch_geometric.data import Data
from .fann import PAIRS

SCHEMA='fann-bounded-graph-v1'


def atomic_npz(path,**arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,suffix='.npz',delete=False) as stream:
        temporary=Path(stream.name);np.savez(stream,**arrays)
    temporary.replace(path)


def csr_from_hexahedra(cells,nodes):
    """Deduplicate actual bidirectional HEXA_8 edges on CPU; no E-by-hidden tensor."""
    count=len(cells)*24
    dtype=np.int32 if nodes<2**31 else np.int64
    rows=np.empty(count,dtype=dtype);cols=np.empty(count,dtype=dtype)
    offset=0
    for start in range(0,len(cells),65536):
        pairs=cells[start:start+65536,PAIRS].reshape(-1,2);size=len(pairs)
        rows[offset:offset+size]=pairs[:,0];cols[offset:offset+size]=pairs[:,1]
        rows[offset+size:offset+size*2]=pairs[:,1];cols[offset+size:offset+size*2]=pairs[:,0]
        offset+=size*2
    adjacency=sparse.coo_matrix((np.ones(count,dtype=bool),(rows,cols)),shape=(nodes,nodes)).tocsr()
    adjacency=adjacency.astype(np.float32);adjacency.data.fill(1.);adjacency.sort_indices()
    return adjacency


def normalize_gcn(adjacency,improved=True):
    # Graph weights are positive: preserve existing loops, add missing ones.
    # CSR addition avoids setdiag's huge COO insertion/sort on full-domain graphs.
    diagonal=adjacency.diagonal()
    fill=np.where(diagonal==0,2. if improved else 1.,0.).astype(np.float32)
    result=(adjacency+sparse.diags(fill,format='csr')).tocsr()
    degree=np.asarray(result.sum(axis=1)).reshape(-1)
    inverse=np.zeros_like(degree);np.power(degree,-.5,out=inverse,where=degree>0)
    result.sort_indices()
    for lo in range(0,result.nnz,1048576):
        hi=min(lo+1048576,result.nnz)
        first=np.searchsorted(result.indptr,lo,side='right')-1
        last=np.searchsorted(result.indptr,hi-1,side='right')-1
        counts=np.diff(result.indptr[first:last+2]).astype(np.int64)
        counts[0]-=lo-result.indptr[first];counts[-1]-=result.indptr[last+1]-hi
        rows=np.repeat(np.arange(first,last+1),counts)
        result.data[lo:hi]*=inverse[rows]*inverse[result.indices[lo:hi]]
    return result


def sparse_chunks(adjacency,chunk,device,dtype):
    for lo in range(0,adjacency.nnz,chunk):
        hi=min(lo+chunk,adjacency.nnz)
        first=np.searchsorted(adjacency.indptr,lo,side='right')-1
        last=np.searchsorted(adjacency.indptr,hi-1,side='right')-1
        counts=np.diff(adjacency.indptr[first:last+2]).astype(np.int64)
        counts[0]-=lo-adjacency.indptr[first];counts[-1]-=adjacency.indptr[last+1]-hi
        rows=np.repeat(np.arange(first,last+1,dtype=np.int64),counts)
        yield (torch.as_tensor(rows,device=device,dtype=torch.long),
               torch.as_tensor(adjacency.indices[lo:hi].astype(np.int64),device=device),
               torch.as_tensor(adjacency.data[lo:hi],device=device,dtype=dtype))


class ConstantSparseMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx,values,adjacency,chunk):
        ctx.adjacency=adjacency;ctx.chunk=chunk
        result=values.new_zeros((adjacency.shape[0],values.shape[1]))
        for rows,cols,weights in sparse_chunks(adjacency,chunk,values.device,values.dtype):
            result.index_add_(0,rows,values[cols]*weights[:,None])
        return result
    @staticmethod
    def backward(ctx,gradient):
        result=gradient.new_zeros((ctx.adjacency.shape[1],gradient.shape[1]))
        for rows,cols,weights in sparse_chunks(ctx.adjacency,ctx.chunk,gradient.device,gradient.dtype):
            result.index_add_(0,cols,gradient[rows]*weights[:,None])
        return result,None,None


class BoundedGraphExecution:
    def __init__(self,name,config,cache_dir):
        self.name=name;self.owner_nodes=int(config.get('graph_owner_nodes',32768))
        self.edge_chunk=int(config.get('graph_edge_chunk',262144))
        self.memory_limit=int(config.get('graph_cache_mb',1024))*1024**2
        if self.owner_nodes<1 or self.edge_chunk<1:raise ValueError('Graph chunk sizes must be positive')
        self.cache_dir=Path(cache_dir);self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.topology=None;self.cells=None;self.key=None;self.parts=OrderedDict();self.part_bytes=0
        self.base_normalized=None;self.base_augmented=None

    def adjacency(self,sample):
        cells=sample['hexahedra'];nodes=len(sample['coordinates'])
        if cells is self.cells and self.topology is not None:return self.topology
        digest=hashlib.sha256((SCHEMA+str((nodes,cells.shape,cells.dtype.str))).encode())
        digest.update(memoryview(np.ascontiguousarray(cells)).cast('B'));key=digest.hexdigest()
        if key!=self.key:
            path=self.cache_dir/(key+'.npz')
            with (self.cache_dir/(key+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if path.exists():
                    adjacency=sparse.load_npz(path)
                    if adjacency.shape!=(nodes,nodes):raise ValueError('Invalid graph cache shape')
                else:
                    print(f'{self.name}: building CPU graph nodes={nodes} cells={len(cells)}',flush=True)
                    adjacency=csr_from_hexahedra(cells,nodes)
                    with tempfile.NamedTemporaryFile(dir=self.cache_dir,suffix='.npz',delete=False) as f:
                        temporary=Path(f.name)
                    sparse.save_npz(temporary,adjacency,compressed=False);temporary.replace(path)
            self.topology=adjacency;self.key=key;self.parts.clear();self.part_bytes=0
            self.base_normalized=None;self.base_augmented=None
            print(f'{self.name}: CPU graph ready nodes={nodes} directed_edges={adjacency.nnz}',flush=True)
        self.cells=cells
        return self.topology

    def partition(self,adjacency,start,hops):
        token=(start,hops)
        if token in self.parts:
            self.parts.move_to_end(token);return self.parts[token]
        end=min(start+self.owner_nodes,adjacency.shape[0])
        path=self.cache_dir/self.key/f'owner_{self.owner_nodes}_halo_{hops}_{start}.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as data:part={k:data[k] for k in ['nodes','owners','src','dst']}
        else:
            owners=np.arange(start,end,dtype=np.int64);nodes=owners
            for _ in range(hops):nodes=np.union1d(nodes,adjacency[nodes].indices)
            local=adjacency[nodes][:,nodes].tocoo()
            part={'nodes':nodes.astype(np.int64),'owners':np.searchsorted(nodes,owners).astype(np.int64),
                  'src':local.col.astype(np.int32),'dst':local.row.astype(np.int32)}
            atomic_npz(path,**part)
        if not np.array_equal(part['nodes'][part['owners']],np.arange(start,end)):
            raise ValueError('Partition owner coverage mismatch')
        size=sum(v.nbytes for v in part.values())
        if size<=self.memory_limit:
            self.parts[token]=part;self.part_bytes+=size
            while self.part_bytes>self.memory_limit:
                _,old=self.parts.popitem(last=False);self.part_bytes-=sum(v.nbytes for v in old.values())
        return part

    def pieces(self,model,norm,sample,device):
        adjacency=self.adjacency(sample)
        hops=model.net.processor_layers if self.name=='GCN' else model.net.num_layers
        x=norm.graph(sample,'cpu',with_edges=False).x
        for start in range(0,len(x),self.owner_nodes):
            part=self.partition(adjacency,start,hops)
            nodes=part['nodes'];src=part['src'].astype(np.int64);dst=part['dst'].astype(np.int64)
            local=x[torch.from_numpy(nodes)]
            delta=local[torch.from_numpy(dst),:3]-local[torch.from_numpy(src),:3]
            graph=Data(x=local,edge_index=torch.from_numpy(np.stack([src,dst])),
                       edge_attr=torch.cat([delta,delta.norm(dim=1,keepdim=True)],dim=1)).to(device)
            yield graph,torch.as_tensor(part['owners'],device=device),part['nodes'][part['owners']]

    def backward_sample(self,model,norm,sample,device,stop_requested=None):
        """All owners once, enough halo, one summed objective; caller steps once."""
        value=0.;denominator=sample['targets'].size
        for graph,owners,global_ids in self.pieces(model,norm,sample,device):
            if stop_requested is not None and stop_requested():return None
            prediction=model(graph)[owners]
            target=torch.as_tensor((sample['targets'][global_ids]-norm.mean)/norm.scale,device=device)
            loss=(prediction-target).square().sum()/denominator
            if not torch.isfinite(loss):raise ValueError('Nonfinite partition loss')
            loss.backward();value+=float(loss.item())
            del prediction,target,loss,graph
        return value

    def predict_cpu(self,model,norm,sample,device):
        prediction=torch.empty((len(sample['coordinates']),6),dtype=torch.float32)
        for graph,owners,global_ids in self.pieces(model,norm,sample,device):
            prediction[torch.from_numpy(global_ids)]=model(graph)[owners].detach().cpu()
        return prediction

    def graph_unet(self,model,norm,sample,device):
        """Global pooling is preserved; only constant graph algebra is CPU/offloaded."""
        network=model.net.net;adjacency=self.adjacency(sample)
        if self.base_normalized is None:self.base_normalized=normalize_gcn(adjacency)
        normalized=self.base_normalized
        x=norm.graph(sample,device,with_edges=False).x
        def convolution(layer,features,matrix):
            result=ConstantSparseMultiply.apply(layer.lin(features),matrix,self.edge_chunk)
            return result+layer.bias if layer.bias is not None else result
        x=network.act(convolution(network.down_convs[0],x,normalized))
        xs=[x];norms=[normalized];perms=[]
        for i in range(1,network.depth+1):
            if i==1 and self.base_augmented is not None:expanded=self.base_augmented
            else:
                print(f'Graph_UNet: CPU adjacency expansion level={i} nodes={adjacency.shape[0]} input_edges={adjacency.nnz}',flush=True)
                augmented=adjacency+sparse.diags((1.-adjacency.diagonal()).astype(np.float32),format='csr')
                expanded=(augmented@augmented).tocsr();del augmented
                expanded.setdiag(0.);expanded.eliminate_zeros();expanded.sort_indices()
                print(f'Graph_UNet: expanded_edges={expanded.nnz} CSR_GiB={(expanded.data.nbytes+expanded.indices.nbytes+expanded.indptr.nbytes)/2**30:.2f}',flush=True)
                if i==1:self.base_augmented=expanded
            pool=network.pools[i-1]
            selection=pool.select(x,torch.zeros(len(x),device=device,dtype=torch.long))
            perm=selection.node_index;score=selection.weight
            x=x[perm]*score[:,None]
            if pool.multiplier!=1:x=x*pool.multiplier
            ids=perm.detach().cpu().numpy()
            adjacency=expanded[ids][:,ids].tocsr();del expanded
            adjacency.sort_indices()
            normalized=normalize_gcn(adjacency,improved=network.down_convs[i].improved)
            x=network.act(convolution(network.down_convs[i],x,normalized))
            if i<network.depth:xs.append(x);norms.append(normalized)
            perms.append(perm)
        del adjacency,normalized
        for i in range(network.depth):
            j=network.depth-1-i;residual=xs[j];up=torch.zeros_like(residual);up[perms[j]]=x
            x=residual+up if network.sum_res else torch.cat([residual,up],dim=-1)
            x=convolution(network.up_convs[i],x,norms[j])
            if i<network.depth-1:x=network.act(x)
        return x
