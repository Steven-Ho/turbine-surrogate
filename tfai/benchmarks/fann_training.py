"""Resumable, geometry-disjoint faNN experiments on explicit sampled or full domains.

One case is moved to the accelerator at a time. Full-domain does not imply
multi-GPU domain parallelism: unsupported memory requirements fail explicitly.
"""
from tfai.benchmarks.fann_logging import print, log_every, should_log

import collections
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
import numpy as np
import torch
from .fann import FIELDS, FannAdapter, Normalizer, load_sample
from tfai.training.optimization import build_lr_scheduler
from tfai.utils.random_utils import capture_rng_state, restore_rng_state, set_global_seed
from tfai.utils.torch_utils import atomic_torch_save

GRAPH_MODELS = {'GCN','Graph_UNet','Graph_Transformer'}
FORMAT_VERSION = 1


class RunLockedError(RuntimeError):
    pass


def atomic_json(data, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class SampleStore:
    def __init__(self, paths, cache_size=1, fingerprints=None, registered_files=None, sample_dependencies=None):
        self.paths=[str(Path(p).resolve()) for p in paths]
        self.cache=collections.OrderedDict();self.limit=max(1,cache_size)
        self.fingerprints=fingerprints
        self.registered_files=registered_files
        self.sample_dependencies=sample_dependencies
    def __len__(self):return len(self.paths)
    def __iter__(self):
        for i in range(len(self)):yield self[i]
    def __getitem__(self,index):
        if index not in self.cache:
            if self.registered_files is not None:
                sample=load_sample(self.paths[index],validate=False)
                # Prepared data are immutable for this run: trust startup registration.
                # Do not compare per-node filesystem metadata during training/evaluation.
                sample['fingerprint']=self.fingerprints[index]
            else:sample=load_sample(self.paths[index])
            if self.fingerprints and sample['fingerprint']!=self.fingerprints[index]:
                raise ValueError('Sample content changed: '+self.paths[index])
            self.cache[index]=sample
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def _inspection_worker_init():
    # Each worker reads/reduces a whole case; prevent nested Torch CPU pools.
    torch.set_num_threads(1)


def _inspect_case(task):
    path,collect_stats=task
    sample=load_sample(path)
    if 'geometry_id' not in sample:raise ValueError('Missing geometry_id')
    summary={'geometry':int(sample['geometry_id'].reshape(-1)[0]),
             'domain':str(sample.get('selection_protocol',np.array('legacy_first'))),
             'has_wall':'wall_distance' in sample,'physics':sample.get('fixed_conditions'),
             'fingerprint':sample['fingerprint'],'nodes':len(sample['coordinates']),
             'cells':len(sample['hexahedra'])}
    if collect_stats:
        from .fann import normalization_summary
        summary['normalization']=normalization_summary(sample)
    return summary


def inspect_splits(split_path, allow_legacy=False, progress=None, collect_normalization=False, workers=1):
    from .fann import NormalizationStats
    if workers<1:raise ValueError('Startup workers must be positive')
    split_path=Path(split_path).resolve();raw=json.loads(split_path.read_text())
    records={};geometries={};seen={};protocol=None;wall=None;fixed=None
    stats=NormalizationStats();pool=None
    if workers>1:
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=get_context('spawn'),initializer=_inspection_worker_init)
    try:
        for split in ['train','valid','test']:
            paths=raw.get(split,[])
            if not paths:raise ValueError('Need nonempty '+split+' split')
            resolved=[str((Path(value) if Path(value).is_absolute() else split_path.parent/value).resolve()) for value in paths]
            records[split]=[];geometries[split]=set()
            if progress is not None:progress(f'faNN inspecting {split}: {len(paths)} samples, workers={workers}')
            tasks=[(path,collect_normalization and split=='train') for path in resolved]
            summaries=pool.map(_inspect_case,tasks,chunksize=1) if pool else map(_inspect_case,tasks)
            for sample_index,(path,sample) in enumerate(zip(resolved,summaries)):
                if progress is not None and sample_index%10==0:
                    progress(f'faNN inspecting {split} {sample_index+1}/{len(paths)}: {path}')
                geometry=sample['geometry'];geometries[split].add(geometry)
                domain=sample['domain'];has_wall=sample['has_wall']
                if protocol is None:protocol=domain;wall=has_wall
                if domain!=protocol or wall!=has_wall:raise ValueError('Mixed domain or input schemas')
                if not allow_legacy and (domain not in ['structured_patches','full'] or not has_wall):
                    raise ValueError('Formal protocol needs wall_distance and structured_patches/full data; reprepare or explicitly use --allow-legacy-inputs')
                physics=sample['physics']
                if physics is not None:
                    if np.asarray(physics).shape!=(4,) or not np.isfinite(physics).all():raise ValueError('Invalid fixed physics')
                    if fixed is None:fixed=np.asarray(physics).tolist()
                    elif not np.array_equal(physics,fixed):raise ValueError('Fixed inlet/gas conditions changed without model input support')
                identity=sample['fingerprint']
                if identity in seen:raise ValueError('Duplicate sample content: '+path+' and '+seen[identity])
                seen[identity]=path
                records[split].append({'path':path,'fingerprint':identity,'geometry_id':geometry,
                                       'nodes':sample['nodes'],'cells':sample['cells']})
                if collect_normalization and split=='train':stats.add(sample['normalization'])
    finally:
        if pool is not None:pool.shutdown(wait=True,cancel_futures=True)
    for left,right in [('train','valid'),('train','test'),('valid','test')]:
        if geometries[left]&geometries[right]:raise ValueError('Geometry leakage between '+left+' and '+right)
    if len(records['train'])<2:raise ValueError('Formal training needs at least two training samples')
    signature=hashlib.sha256(json.dumps({'splits':records,'domain':protocol,'wall_distance':wall,'fixed_conditions':fixed},sort_keys=True).encode()).hexdigest()
    manifest={'splits':records,'signature':signature,'domain':protocol,'wall_distance':wall,
            'geometry_split':{k:sorted(v) for k,v in geometries.items()},'fixed_conditions':fixed,
            'scope':'full-domain nodal evaluation' if protocol=='full' else 'selected patches only; not full-flow-field accuracy'}
    if collect_normalization:manifest['training_normalization']=stats.state()
    return manifest


def store_for(manifest,split,cache_size=1):
    rows=manifest['splits'][split]
    return SampleStore([r['path'] for r in rows],cache_size,[r['fingerprint'] for r in rows],
                       manifest.get('registered_files'),manifest.get('sample_dependencies'))


def fit_streaming_pod(train,norm,rank,directory,chunk=65536,progress=None):
    """Method of snapshots using disk-backed fields; no full snapshot stack in RAM."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    ref=train[0];nodes=len(ref['coordinates']);dofs=nodes*6;cases=len(train)
    rank=min(rank,cases,dofs)
    required=4*dofs*(cases+rank)
    if shutil.disk_usage(directory).free < required*1.2:
        raise ValueError(f'Insufficient POD scratch space: need about {required/1e9:.1f} GB plus margin')
    snapshots=np.memmap(directory/'snapshots.f32',mode='w+',dtype='float32',shape=(cases,dofs))
    for i,sample in enumerate(train):
        if progress is not None and i%10==0:progress(f'POD snapshots {i+1}/{cases}')
        if not np.array_equal(sample['original_node_ids'],ref['original_node_ids']) or (sample['hexahedra'] is not ref['hexahedra'] and not np.array_equal(sample['hexahedra'],ref['hexahedra'])):
            raise ValueError('POD needs corresponding node IDs and topology')
        for start in range(0,nodes,chunk):
            end=min(start+chunk,nodes)
            snapshots[i,start*6:end*6]=((sample['targets'][start:end]-norm.mean)/norm.scale).reshape(-1)
    snapshots.flush();gram=np.zeros((cases,cases),dtype=np.float64)
    for start in range(0,dofs,chunk*6):
        if progress is not None and (start//(chunk*6))%10==0:progress(f'POD Gram {start}/{dofs} field entries')
        block=np.asarray(snapshots[:,start:start+chunk*6],dtype=np.float64)
        gram+=block@block.T
    values,vectors=np.linalg.eigh(gram);order=np.argsort(values)[::-1]
    keep=order[values[order]>max(float(values.max()),1.)*1e-12][:rank]
    if not len(keep):raise ValueError('POD training snapshots have zero numerical rank')
    transform=vectors[:,keep]/np.sqrt(values[keep])[None,:]
    basis=np.memmap(directory/'basis.f32',mode='w+',dtype='float32',shape=(dofs,len(keep)))
    for start in range(0,dofs,chunk*6):
        if progress is not None and (start//(chunk*6))%10==0:progress(f'POD basis {start}/{dofs} field entries')
        basis[start:start+chunk*6]=snapshots[:,start:start+chunk*6].T@transform
    basis.flush();del snapshots
    (directory/'snapshots.f32').unlink()
    return {'basis':torch.from_numpy(basis),'node_ids':torch.from_numpy(ref['original_node_ids'].astype(np.int64))}


def build_model(name,train,norm,config,state=None,scratch=None):
    pod_state=None
    if name=='POD':
        if state is not None:pod_state={'basis':state['basis'],'node_ids':state['node_ids']}
        elif config.get('streaming_pod',False):
            cache=os.environ.get('TURBINE_FANN_POD_CACHE','runs/fann_pod_cache_v1')
            if cache and cache!='0':
                from .fann_pod_cache import shared_pod_basis
                pod_state=shared_pod_basis(train,norm,config['rank'],cache,scratch)
            else:pod_state=fit_streaming_pod(train,norm,config['rank'],scratch)
    model=FannAdapter(name,train,norm,scale=config['scale'],grid=config['grid'],rank=config['rank'],
                      pod_state=pod_state,pod_geometry_bins=config['pod_geometry_bins'])
    if name=='UNet':
        model.unet_mapping_cache=os.environ.get('TURBINE_FANN_UNET_CACHE','runs/fann_unet_mapping_cache_v1')
    if name=='Transolver_3':
        chunk=int(os.environ.get('TURBINE_FANN_TRANSOLVER_CHUNK_SIZE','8192'))
        if chunk<1:raise ValueError('Transolver chunk size must be positive')
        model.net.chunk_size=chunk
        model.net.use_checkpoint=os.environ.get('TURBINE_FANN_TRANSOLVER_CHECKPOINT','1')=='1'
        print(f'Transolver_3: chunk={chunk} checkpoint={model.net.use_checkpoint}',flush=True)
    if name=='GeoTransolver':
        model.amp_mode=config.get('amp','off')
        model.net.activation_offload=os.environ.get('TURBINE_GEO_ACTIVATION_OFFLOAD','1')=='1'
        chunk=int(os.environ.get('TURBINE_GEO_NODE_CHUNK','32768'))
        if chunk<0:raise ValueError('TURBINE_GEO_NODE_CHUNK must be nonnegative')
        for module in model.net.model.modules():
            module.node_chunk_size=chunk
            if hasattr(module,'index_cache') and module.index_cache.directory is None and 'TURBINE_GEO_CACHE_DIR' not in os.environ:
                module.index_cache.directory=Path('runs/fann_neighbor_cache').resolve()
    if state is not None:model.load_state_dict(state,strict=True)
    if name in GRAPH_MODELS and config.get('graph_execution','dense')=='bounded':
        from .fann_graph_execution import BoundedGraphExecution
        model.graph_executor=BoundedGraphExecution(name,config,
            os.environ.get('TURBINE_FANN_GRAPH_CACHE','runs/fann_graph_topology_cache_v1'))
    return model


def prepare_input(model,norm,sample,device):
    graph=norm.graph(sample,device,with_edges=model.name in GRAPH_MODELS)
    return model.prepare(graph,sample['original_node_ids'])


def evaluate(model,norm,store,device,output=None,near_wall=0.001,stop_requested=None,return_statistics=False):
    """Node-weighted six-channel metrics in physical units; no target projection."""
    model.eval();sse=np.zeros(6);absolute=np.zeros(6);total=np.zeros(6);square=np.zeros(6)
    count=0;normalized_sse=0.;near_sse=np.zeros(6);near_count=0;rows=[]
    if output is not None:Path(output).mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for i,sample in enumerate(store):
            if stop_requested is not None and stop_requested():return None
            executor=getattr(model,'graph_executor',None)
            if executor is not None:
                prediction=(executor.graph_unet(model,norm,sample,device) if model.name=='Graph_UNet'
                            else executor.predict_cpu(model,norm,sample,device))
            else:prediction=model(prepare_input(model,norm,sample,device))
            if tuple(prediction.shape)!=sample['targets'].shape or not torch.isfinite(prediction).all():
                raise ValueError('Invalid validation/test prediction')
            normalized_sse+=float((prediction-norm.target(sample,prediction.device)).double().square().sum().item())
            physical=norm.decode(prediction);truth=sample['targets'].astype(np.float64)
            error=physical.astype(np.float64)-truth;n=len(truth)
            sse+=(error**2).sum(0);absolute+=np.abs(error).sum(0)
            total+=truth.sum(0);square+=(truth**2).sum(0);count+=n
            mask=sample.get('wall_distance',np.full(n,np.inf))<near_wall
            near_sse+=(error[mask]**2).sum(0);near_count+=int(mask.sum())
            row={'source':sample['source'],'geometry_id':int(sample['geometry_id'].reshape(-1)[0]),
                 'nodes':n,'rmse':np.sqrt((error**2).mean(0)).tolist(),'mae':np.abs(error).mean(0).tolist()}
            if output is not None:
                path=Path(output)/f'prediction_{i:05d}.npz'
                with tempfile.NamedTemporaryFile(dir=output,suffix='.npz',delete=False) as stream:
                    temporary=Path(stream.name)
                    np.savez_compressed(stream,original_node_ids=sample['original_node_ids'],prediction=physical,
                                        output_names=np.array(FIELDS),source=np.array(sample['source']))
                temporary.replace(path);row['prediction_file']=str(path.resolve())
            rows.append(row)
            del prediction
    denominator=square-total**2/count
    r2=[float(1-sse[i]/denominator[i]) if denominator[i]>0 else None for i in range(6)]
    result={'normalized_mse':normalized_sse/(count*6),'nodes':count,'cases':len(store),'fields':list(FIELDS),
            'units':['m/s','m/s','m/s','Pa','K','kg/m^3'],'rmse':np.sqrt(sse/count).tolist(),
            'mae':(absolute/count).tolist(),'r2':r2,
            'near_wall':{'threshold_m':near_wall,'nodes':near_count,'rmse':np.sqrt(near_sse/near_count).tolist() if near_count else None},
            'samples':rows}
    if return_statistics:
        result['statistics']={'sse':sse.tolist(),'absolute':absolute.tolist(),'total':total.tolist(),
            'square':square.tolist(),'normalized_sse':normalized_sse,'count':count,
            'near_sse':near_sse.tolist(),'near_count':near_count}
    return result


def assert_resume_config(old,new):
    old=dict(old)
    # Accept graph checkpoints written with an accidental tuple configuration key.
    for key in list(old):
        if isinstance(key,tuple) and key and key[0]=='graph_execution_schema':
            old.setdefault('graph_execution_schema',old.pop(key))
    new=dict(new)
    for key,default in [('beta1',0.9),('beta2',0.999)]:
        old.setdefault(key,default);new.setdefault(key,default)
    immutable=['beta1','beta2','seed','learning_rate','lr_patience','lr_factor','lr_threshold','lr_cooldown','min_lr',
               'patience','grid','rank','scale','pod_geometry_bins','streaming_pod','grad_clip','near_wall','geo_neighbor_policy','graph_execution','graph_owner_nodes','graph_edge_chunk','graph_execution_schema','amp','sampler']
    mismatches=[key for key in immutable if old.get(key)!=new.get(key)]
    if mismatches:raise ValueError('Resume configuration changed: '+','.join(mismatches))


def train_one(name,manifest,config,folder,device,resume=False,max_updates=None):
    import fcntl
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RunLockedError('Another process holds '+str(folder))
        return _train_one(name,manifest,config,folder,device,resume,max_updates)


def execution_config(name,config):
    config=dict(config)
    if config.get('sampler','random')=='random':config.pop('sampler',None)
    elif config['sampler']!='geometry_affinity' or name!='GeoTransolver':raise ValueError('Geometry affinity currently supports GeoTransolver DDP only')
    if config.get('amp','off')=='off':config.pop('amp',None)
    elif config['amp']!='bf16' or name!='GeoTransolver':raise ValueError('BF16 AMP currently supports GeoTransolver only')
    if name not in GRAPH_MODELS:
        config={key:value for key,value in config.items() if not key.startswith('graph_')}
    if name in GRAPH_MODELS and config.get('graph_execution','dense')=='bounded':
        from .fann_graph_execution import SCHEMA as graph_schema
        config['graph_execution_schema']=graph_schema
    if name=='GeoTransolver':
        from tfai.models._vendor.geotransolver2.spatial_neighbors import SCHEMA
        config['geo_neighbor_policy']={'schema':SCHEMA,'backend':os.environ.get('TURBINE_GEO_QUERY_BACKEND','kdtree'),
            'reference_max_points':int(os.environ.get('TURBINE_GEO_REFERENCE_MAX_POINTS','32768'))}
    return config


def _train_one(name,manifest,config,folder,device,resume=False,max_updates=None):
    interval=log_every()
    config=execution_config(name,config)
    last_path=folder/'last.pt';best_path=folder/'best.pt'
    if not resume and (last_path.exists() or best_path.exists()):raise ValueError('Existing run; use --resume or a new output directory')
    if resume and not last_path.exists():raise ValueError('No last.pt to resume')
    saved=torch.load(last_path,map_location='cpu',weights_only=False) if resume else None
    if saved:
        if saved.get('format_version')!=FORMAT_VERSION or saved['model']!=name:raise ValueError('Wrong checkpoint format/model')
        if saved['manifest']['signature']!=manifest['signature']:raise ValueError('Dataset/split contents changed')
        assert_resume_config(saved['config'],config)
        if saved['completed_epochs']>0 and not best_path.exists():raise ValueError('Missing validation-best checkpoint')
    set_global_seed(config['seed'])
    train=store_for(manifest,'train',config['cache_samples']);valid=store_for(manifest,'valid',config['cache_samples'])
    print(f'{name}: '+('restoring normalization' if saved else ('using registered normalization' if 'training_normalization' in manifest else f'fitting normalization over {len(train)} training samples')),flush=True)
    if saved:norm=Normalizer.from_state(saved['normalization'])
    elif 'training_normalization' in manifest:
        norm=Normalizer.from_state(manifest['training_normalization'])
        print(f'{name}: reusing shared training-only normalization; no second dataset pass',flush=True)
    else:norm=Normalizer(train,progress=lambda message:print(name+': '+message,flush=True))
    print(f'{name}: normalization ready; building model',flush=True)
    model=build_model(name,train,norm,config,state=saved['state_dict'] if saved else None,scratch=folder/'pod_scratch').to(device)
    from .fann_model_parallel import configure
    configure(model,device)
    print(f'{name}: model ready; entering training loop',flush=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=(config.get('beta1',0.9),config.get('beta2',0.999)))
    scheduler=build_lr_scheduler(optimizer,{'lr_scheduler':{'enabled':True,'factor':config['lr_factor'],
        'patience':config['lr_patience'],'threshold':config['lr_threshold'],'cooldown':config['lr_cooldown'],'min_lr':config['min_lr']}})
    epoch=0;position=0;global_step=0;best=math.inf;bad=0;history=[];pending_sum=0.;pending_count=0;elapsed=0.
    if saved:
        optimizer.load_state_dict(saved['optimizer_state_dict']);scheduler.load_state_dict(saved['scheduler_state_dict'])
        epoch=saved['completed_epochs'];position=saved['position'];global_step=saved['global_step']
        best=saved['best_score'];bad=saved['early_stopping_counter'];history=saved['history']
        pending_sum=saved['pending_loss_sum'];pending_count=saved['pending_elements'];elapsed=saved.get('training_seconds',0.)
        restore_rng_state(saved['rng_state'])
    profile_steps=int(os.environ.get('TURBINE_FANN_PROFILE_STEPS','0'))
    if profile_steps<0:raise ValueError('TURBINE_FANN_PROFILE_STEPS must be nonnegative')
    stopped=False;interrupted=False;started=time.perf_counter();updates=0
    def checkpoint():
        return {'format_version':FORMAT_VERSION,'model':name,'state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
            'normalization':vars(norm),'config':config,'manifest':manifest,'fields':list(FIELDS),
            'completed_epochs':epoch,'position':position,'global_step':global_step,'best_score':best,
            'early_stopping_counter':bad,'history':history,'pending_loss_sum':pending_sum,'pending_elements':pending_count,
            'rng_state':capture_rng_state(),'training_seconds':elapsed+time.perf_counter()-started,
            'execution_runtime':{'pipeline_devices':[str(d) for d in getattr(model,'pipeline_devices',[device])],
                'geo_activation_offload':getattr(model.net,'activation_offload',False),
                'geo_node_chunk':getattr(getattr(model.net,'model',None),'node_chunk_size',None),
                'transolver_chunk':getattr(model.net,'chunk_size',None),
                'transolver_checkpoint':getattr(model.net,'use_checkpoint',None)}}
    old_handlers={}
    def stop(signum,frame):
        nonlocal interrupted
        interrupted=True
    for sig in [signal.SIGTERM,signal.SIGINT]:
        old_handlers[sig]=signal.signal(sig,stop)
    try:
        atomic_json({'state':'training','model':name,'pid':os.getpid(),'completed_epochs':epoch,'global_step':global_step,
                     'scope':manifest['scope']},folder/'status.json')
        case_checkpoint=(getattr(model,'graph_executor',None) is not None or name=='GeoTransolver')
        if case_checkpoint and saved is None:
            atomic_torch_save(checkpoint(),last_path)
        while epoch<config['epochs'] and bad<config['patience']:
            generator=torch.Generator().manual_seed(config['seed']+epoch)
            order=torch.randperm(len(train),generator=generator).tolist();model.train()
            for offset in range(position,len(order)):
                if interrupted or (max_updates is not None and updates>=max_updates):
                    interrupted=True;break
                profile=updates<profile_steps
                phase_times={}
                def profile_sync():
                    if profile and str(device).startswith('cuda'):
                        for d in getattr(model,'pipeline_devices',[device]):torch.cuda.synchronize(d)
                profile_sync();phase_started=time.perf_counter()
                def mark(phase):
                    nonlocal phase_started
                    if profile:
                        profile_sync();now=time.perf_counter()
                        phase_times[phase]=now-phase_started;phase_started=now
                batch_started=time.perf_counter()
                sample=train[order[offset]];optimizer.zero_grad(set_to_none=True)
                mark('load_sample')
                executor=getattr(model,'graph_executor',None)
                if executor is not None and name in ['GCN','Graph_Transformer']:
                    case_rng=capture_rng_state()
                    loss_value=executor.backward_sample(model,norm,sample,device,stop_requested=lambda:interrupted)
                    if loss_value is None:
                        optimizer.zero_grad(set_to_none=True);restore_rng_state(case_rng)
                        interrupted=True;break
                    elements=sample['targets'].size
                    mark('input_forward_backward')
                else:
                    if executor is not None:
                        prediction=executor.graph_unet(model,norm,sample,device)
                        mark('input_forward')
                    else:
                        inputs=prepare_input(model,norm,sample,device);mark('prepare_input')
                        prediction=model(inputs);del inputs;mark('forward')
                    target=norm.target(sample,device)
                    if prediction.shape!=target.shape:raise ValueError('Output shape mismatch')
                    loss=(prediction-target).square().mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    mark('target_loss')
                    loss.backward();loss_value=float(loss.item());elements=target.numel()
                    mark('backward')
                    del prediction,target,loss
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();global_step+=1;updates+=1;position=offset+1
                mark('clip_optimizer')
                pending_sum+=loss_value*elements;pending_count+=elements
                if case_checkpoint:atomic_torch_save(checkpoint(),last_path)
                if should_log(position,len(order),interval):
                    memory=(sum(torch.cuda.memory_allocated(d) for d in getattr(model,'pipeline_devices',[device]))/2**30 if str(device).startswith('cuda') else 0.)
                    print(f'{name} epoch={epoch+1}/{config["epochs"]} batch={position}/{len(order)} step={global_step} loss={loss_value:.7g} train={pending_sum/pending_count:.7g} lr={optimizer.param_groups[0]["lr"]:.4g} batch_seconds={time.perf_counter()-batch_started:.3f} allocated_GiB={memory:.3f}')
                if profile:
                    mark('checkpoint')
                    print(f'{name} step_profile step={global_step} nodes={len(sample["coordinates"])} seconds='+json.dumps(phase_times),flush=True)
            if interrupted:
                atomic_torch_save(checkpoint(),last_path)
                atomic_json({'state':'interrupted','model':name,'completed_epochs':epoch,'position':position,
                             'global_step':global_step},folder/'status.json')
                return {'state':'interrupted','global_step':global_step}
            metrics=evaluate(model,norm,valid,device,near_wall=config['near_wall'],stop_requested=lambda:interrupted)
            if metrics is None:
                atomic_torch_save(checkpoint(),last_path)
                atomic_json({'state':'interrupted','model':name,'completed_epochs':epoch,'position':position,
                             'global_step':global_step,'during':'validation'},folder/'status.json')
                return {'state':'interrupted','global_step':global_step}
            value=metrics['normalized_mse'];scheduler.step(value)
            epoch+=1;position=0
            improved=value<best
            if improved:best=value;bad=0
            else:bad+=1
            history.append({'epoch':epoch,'global_step':global_step,'train_loss':pending_sum/pending_count,
                            'val_loss':value,'lr':optimizer.param_groups[0]['lr'],'validation_rmse':metrics['rmse']})
            pending_sum=0.;pending_count=0
            if improved:atomic_torch_save(checkpoint(),best_path)
            atomic_torch_save(checkpoint(),last_path)
            atomic_json(history,folder/'history.json')
            atomic_json({'state':'training','model':name,'pid':os.getpid(),'completed_epochs':epoch,'global_step':global_step,
                'best_validation_loss':best,'early_stopping_counter':bad,'lr':optimizer.param_groups[0]['lr'],
                'latest_validation_rmse':metrics['rmse'],'scope':manifest['scope']},folder/'status.json')
            print(f'{name} epoch={epoch}/{config["epochs"]} step={global_step} train={history[-1]["train_loss"]:.7g} val={value:.7g} lr={optimizer.param_groups[0]["lr"]:.4g} bad={bad}/{config["patience"]}',flush=True)
        stopped=bad>=config['patience']
    finally:
        for sig,handler in old_handlers.items():signal.signal(sig,handler)
    training_seconds=elapsed+time.perf_counter()-started
    # Freeze the validation-selected checkpoint before loading any test samples for metrics.
    best_saved=torch.load(best_path,map_location='cpu',weights_only=False)
    model.load_state_dict(best_saved['state_dict'],strict=True)
    atomic_json({'state':'evaluating','model':name,'completed_epochs':epoch,'best_epoch':best_saved['completed_epochs']},folder/'status.json')
    digest=file_hash(best_path)
    final_path=folder/'report.json'
    if final_path.exists():
        previous=json.loads(final_path.read_text())
        if previous.get('best_checkpoint_sha256')==digest and previous.get('dataset_signature')==manifest['signature']:
            previous.update(completed_epochs=epoch,optimizer_steps=global_step,early_stopped=stopped,config=config,training_seconds=training_seconds)
            atomic_json(previous,final_path);atomic_json({'state':'complete','model':name,'completed_epochs':epoch},folder/'status.json')
            return previous
    validation=evaluate(model,norm,valid,device,output=folder/'predictions/valid',near_wall=config['near_wall'])
    test=evaluate(model,norm,store_for(manifest,'test',config['cache_samples']),device,output=folder/'predictions/test',near_wall=config['near_wall'])
    report={'state':'complete','model':name,'completed_epochs':epoch,'best_epoch':best_saved['completed_epochs'],
        'optimizer_steps':global_step,'early_stopped':stopped,'best_checkpoint_sha256':digest,
        'dataset_signature':manifest['signature'],'scope':manifest['scope'],'validation':validation,'test':test,
        'training_seconds':training_seconds,'fields':list(FIELDS),
        'selection':'minimum complete-validation normalized MSE; test evaluated only after selection',
        'config':config,'environment':{'torch':torch.__version__,'device':str(device),
            'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,
            'geo_query_backend':os.environ.get('TURBINE_GEO_QUERY_BACKEND','default')}}
    atomic_json(report,final_path);atomic_json({'state':'complete','model':name,'completed_epochs':epoch,
        'best_epoch':best_saved['completed_epochs'],'early_stopped':stopped},folder/'status.json')
    return report


def evaluate_frozen(path,manifest,split,device,output):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('format_version')!=FORMAT_VERSION:raise ValueError('Not a formal faNN checkpoint')
    if checkpoint['manifest']['signature']!=manifest['signature']:raise ValueError('Dataset/split changed')
    norm=Normalizer.from_state(checkpoint['normalization'])
    model=build_model(checkpoint['model'],[],norm,checkpoint['config'],state=checkpoint['state_dict']).to(device)
    from .fann_model_parallel import configure
    configure(model,device)
    result=evaluate(model,norm,store_for(manifest,split),device,output=Path(output)/'predictions',near_wall=checkpoint['config']['near_wall'])
    result.update(model=checkpoint['model'],split=split,checkpoint_sha256=file_hash(path),scope=manifest['scope'])
    atomic_json(result,Path(output)/'report.json');return result
