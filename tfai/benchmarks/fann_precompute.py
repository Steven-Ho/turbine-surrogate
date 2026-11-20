"""Build data/config-dependent model artifacts; never cache learned predictions."""
from datetime import datetime,timezone
import fcntl
import csv
import io
import hashlib
import json
import os
import shutil
from pathlib import Path
import time
import numpy as np
import torch
from .fann import MODELS,FannAdapter,Normalizer,voxel_queries,_load_shared_mesh
from .fann_training import atomic_json,store_for
from .fann_startup_cache import file_stamp,verify_files
from .fann_graph_execution import BoundedGraphExecution
from .fann_pod_cache import shared_pod_basis
from .fann_unet_cache import nearest_mapping,mapping_path
from tfai.models._vendor.geotransolver2.ball_query import BQWarp


def timestamp():return datetime.now(timezone.utc).isoformat()


def read_coordinates(path,manifest):
    deps={p:manifest['registered_files'][p] for p in manifest['sample_dependencies'][path]} if 'registered_files' in manifest else {path:file_stamp(path)}
    verify_files(deps)
    with np.load(path,allow_pickle=False) as data:coordinates=np.ascontiguousarray(data['coordinates'])
    verify_files(deps)
    return coordinates


def geometry_index(manifest,home,progress):
    path=home/('geometry_'+manifest['signature']+'.json')
    if path.exists():
        index=json.loads(path.read_text())
        if index['signature']==manifest['signature'] and index['schema']=='fann-geometry-index-v1':
            progress('Geometry index hit: '+str(path));return index
    groups={};sources={};rows=[row for split in ('train','valid','test') for row in manifest['splits'][split]]
    for i,row in enumerate(rows):
        coordinates=read_coordinates(row['path'],manifest)
        digest=hashlib.sha256(str((coordinates.shape,coordinates.dtype.str)).encode())
        digest.update(memoryview(coordinates).cast('B'));key=digest.hexdigest()
        groups.setdefault(key,{'source':row['path'],'nodes':len(coordinates)});sources[row['path']]=key
        if i%10==0:progress(f'Geometry index {i+1}/{len(rows)}; distinct={len(groups)}')
    index={'schema':'fann-geometry-index-v1','signature':manifest['signature'],'geometries':groups,'sources':sources}
    atomic_json(index,path);return index


def topology_samples(manifest):
    """Read each shared mesh once. Inline meshes are deduplicated by CSR keys."""
    seen=set()
    for split in ('train','valid','test'):
        for row in manifest['splits'][split]:
            path=row['path']
            with np.load(path,allow_pickle=False) as data:
                mesh=Path(str(data['mesh_file'])) if 'mesh_file' in data.files else None
                if mesh is None:cells=data['hexahedra']
            if mesh is not None:
                if not mesh.is_absolute():mesh=Path(path).parent/mesh
                token=(str(mesh.resolve()),row['nodes'])
                if token in seen:continue
                seen.add(token);s=mesh.stat()
                cells=_load_shared_mesh(str(mesh.resolve()),(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))['hexahedra']
            # Only the node count is needed to construct adjacency; no fields loaded.
            yield {'coordinates':np.empty((row['nodes'],0),np.float32),'hexahedra':cells}


class CacheReport:
    def __init__(self,output,manifest,config,device):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.started=time.perf_counter();self.artifacts={}
        self.data={'state':'running','started_utc':timestamp(),'dataset_signature':manifest['signature'],
                   'config':config,'device':str(device),'models':{},'shared_phases':{},
                   'not_cacheable':['Fresh NN parameter initialization and checkpoint transfer',
                       'Current sample reading/decompression and input/device transfer',
                       'Learned activations, attention/slice states and predictions',
                       'Graph_UNet deeper pooled adjacency depends on weights and must be recomputed',
                       'DDP communicator initialization']}
        self.write()
    def write(self):
        self.data['elapsed_seconds']=time.perf_counter()-self.started
        self.data['unique_artifact_bytes']=sum(self.artifacts.values())
        self.data['total_seconds']=self.data['elapsed_seconds']+self.data['shared_phases'].get('registration_seconds',0.)
        atomic_json(self.data,self.output/'report.json')
        stream=io.StringIO();columns=['model','state','wall_seconds','task_seconds','units','built_tasks','cache_hits']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for name,entry in self.data['models'].items():writer.writerow({'model':name,**{key:entry.get(key,'') for key in columns[1:]}})
        temporary=self.output/'summary.csv.tmp';temporary.write_text(stream.getvalue());temporary.replace(self.output/'summary.csv')
    def unit(self,model,kind,start,paths=(),status='ready',**extra):
        files=[]
        for path in paths:
            path=Path(path)
            if path.exists():
                size=path.stat().st_size;self.artifacts[str(path.resolve())]=size
                files.append({'path':str(path.resolve()),'bytes':size})
        record={'run_started_utc':self.data['started_utc'],'model':model,'kind':kind,'seconds':time.perf_counter()-start,'status':status,'files':files,**extra}
        entry=self.data['models'][model];entry['units']+=1
        entry['task_seconds']+=record['seconds'];entry['cache_hits']+=int(status=='hit');entry['built_tasks']+=int(status=='built')
        with (self.output/'timings.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        self.write()
    def model_start(self,name):
        self.data['models'][name]={'state':'running','units':0,'task_seconds':0.,'cache_hits':0,'built_tasks':0}
        self.write();print(f'Cache [{name}] start',flush=True)
    def model_end(self,name,start):
        self.data['models'][name].update(state='complete',wall_seconds=time.perf_counter()-start)
        self.write();print(f'Cache [{name}] complete seconds={time.perf_counter()-start:.3f}',flush=True)


def precompute(manifest,models,config,output,device='cuda',pod_checkpoint=None,startup_seconds=0.):
    if not models or any(name not in MODELS for name in models):raise ValueError('Select supported model names')
    device=torch.device(device);home=Path(os.environ.get('TURBINE_FANN_CACHE_WORK_ROOT','runs/fann_cache_work_v1'))
    home.mkdir(parents=True,exist_ok=True)
    report=CacheReport(output,manifest,config,device)
    report.data['requested_models']=list(models)
    report.data['cpu_threads']=torch.get_num_threads()
    report.data['spatial_workers']={kind:os.environ.get(variable,'1') for kind,variable in [('unet','TURBINE_FANN_TREE_WORKERS'),('geo','TURBINE_GEO_TREE_WORKERS')]}
    report.data['cache_work_free_bytes_at_start']=shutil.disk_usage(home).free
    report.data['shared_phases']['registration_seconds']=startup_seconds
    report.data['torch_version']=str(torch.__version__)
    report.data['device_name']=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'
    with (home/'precompute.lock').open('a') as lock:
        waiting=time.perf_counter()
        print('Waiting for exclusive model-cache preparation lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
        report.data['shared_phases']['cache_lock_wait_seconds']=time.perf_counter()-waiting
        try:
            norm=Normalizer.from_state(manifest['training_normalization'])
            index=None
            if any(name in models for name in ('UNet','GeoTransolver')):
                start=time.perf_counter();index=geometry_index(manifest,home,lambda s:print(s,flush=True))
                report.data['shared_phases']['geometry_index_seconds']=time.perf_counter()-start
                report.data['geometry_count']=len(index['geometries'])
                report.data['estimated_geo_neighbor_bytes']=sum(g['nodes']*(min(8,g['nodes'])+min(32,g['nodes']))*4 for g in index['geometries'].values())
                report.write()
            for name in models:
                started=time.perf_counter();report.model_start(name)
                if name=='POD':
                    cache=os.environ.get('TURBINE_FANN_POD_CACHE','runs/fann_pod_cache_v1')
                    if not cache or cache=='0':raise ValueError('POD disk cache must be enabled for precomputation')
                    state=shared_pod_basis(store_for(manifest,'train'),norm,config['rank'],cache,checkpoint=pod_checkpoint)
                    report.unit(name,'training-only POD basis',started,[state['cache_path']],state['cache_status'],basis_shape=list(state['basis'].shape));del state
                elif name in ('GCN','Graph_UNet','Graph_Transformer'):
                    model=FannAdapter(name,[],norm,scale=config['scale'])
                    root=os.environ.get('TURBINE_FANN_GRAPH_CACHE','runs/fann_graph_topology_cache_v1')
                    visited=set()
                    owners=config['owner_nodes'][:1] if name=='Graph_UNet' else config['owner_nodes']
                    for owner in owners:
                        execution=BoundedGraphExecution(name,{'graph_owner_nodes':owner,'graph_edge_chunk':config['graph_edge_chunk'],'graph_cache_mb':config['graph_cache_mb']},root)
                        for sample in topology_samples(manifest):
                            start=time.perf_counter();adjacency=execution.adjacency(sample);key=execution.key
                            token=(key,owner if name!='Graph_UNet' else None)
                            if token in visited:continue
                            visited.add(token);report.unit(name,'CSR topology',start,[Path(root)/(key+'.npz')],status='built' if execution.last_topology_source=='built' else 'hit')
                            if name=='Graph_UNet':
                                paths=[Path(root)/key/'unet_base_normalized_v1.npz',Path(root)/key/'unet_base_squared_v1.npz']
                                ready=all(path.exists() for path in paths);start=time.perf_counter()
                                execution.prepare_unet_static(sample)
                                report.unit(name,'static base normalization and adjacency square',start,paths,'hit' if ready else 'built')
                            else:
                                hops=model.net.processor_layers if name=='GCN' else model.net.num_layers
                                for start_node in range(0,adjacency.shape[0],owner):
                                    path=Path(root)/key/f'owner_{owner}_halo_{hops}_{start_node}.npz';ready=path.exists();start=time.perf_counter()
                                    execution.partition(adjacency,start_node,hops)
                                    report.unit(name,'owner+halo',start,[path],'hit' if ready else 'built',owner_nodes=owner,hops=hops,start_node=start_node)
                                    if (start_node//owner)%20==0:print(f'Cache [{name}] owner={owner} nodes={start_node}/{adjacency.shape[0]}',flush=True)
                        del execution
                    del model
                elif name=='UNet':
                    root=os.environ.get('TURBINE_FANN_UNET_CACHE','runs/fann_unet_mapping_cache_v1')
                    if not root:raise ValueError('UNet disk cache must be enabled')
                    for i,geometry in enumerate(index['geometries'].values()):
                        start=time.perf_counter();coordinates=read_coordinates(geometry['source'],manifest)
                        xyz=torch.as_tensor(((coordinates-norm.origin)/norm.length).astype(np.float32),device=device)
                        positions,_,_=voxel_queries(xyz,config['grid'])
                        p=xyz.cpu().numpy();q=positions.cpu().numpy().reshape(-1,3)
                        path=mapping_path(p,q,root);ready=path.exists();distance,ids=nearest_mapping(p,q,root)
                        report.unit(name,'voxel nearest mapping',start,[path],'hit' if ready else 'built',source=geometry['source'])
                        del coordinates,xyz,positions,p,q,distance,ids
                        print(f'Cache [{name}] geometry {i+1}/{len(index["geometries"])}',flush=True)
                elif name=='GeoTransolver':
                    if os.environ.get('TURBINE_GEO_QUERY_BACKEND','kdtree')!='kdtree':raise ValueError('Geo disk preparation requires kdtree backend')
                    root=os.environ.get('TURBINE_GEO_CACHE_DIR','runs/fann_neighbor_cache')
                    if not root:raise ValueError('Geo disk cache must be enabled')
                    from tfai.models._vendor.geotransolver2.spatial_neighbors import NeighborIndexCache
                    queries=[BQWarp(radius=r,neighbors_in_radius=k,backend='kdtree') for r,k in [(.05,8),(.25,32)]]
                    for query in queries:query.index_cache=NeighborIndexCache(root,0)
                    for i,geometry in enumerate(index['geometries'].values()):
                        coordinates=read_coordinates(geometry['source'],manifest)
                        xyz=torch.as_tensor(((coordinates-norm.origin)/norm.length).astype(np.float32),device=device)
                        for query in queries:
                            print(f'Cache [{name}] geometry {i+1}/{len(index["geometries"])} radius={query.radius} neighbors={query.neighbors_in_radius}',flush=True)
                            start=time.perf_counter();before=query.tree_builds;raw=query._spatial_raw(xyz,xyz)
                            key=query.last_cache_key;path=Path(root)/key[:2]/(key+'.npz')
                            report.unit(name,'radius neighbors',start,[path],'built' if query.tree_builds>before else 'hit',
                                        radius=query.radius,neighbors=query.neighbors_in_radius,source=geometry['source']);del raw
                        del coordinates,xyz
                        print(f'Cache [{name}] geometry {i+1}/{len(index["geometries"])}',flush=True)
                    del queries
                elif name=='Transolver_3':
                    report.unit(name,'no additional static geometry cache',started,status='not_required')
                report.model_end(name,started)
            if 'registered_files' in manifest:verify_files(manifest['registered_files'])
            report.data.update(state='complete',finished_utc=timestamp());report.write()
        except BaseException as error:
            report.data.update(state='failed',error=str(error),finished_utc=timestamp());report.write();raise
    return report.data
