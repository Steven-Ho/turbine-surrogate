"""One process per GPU, complete-case DDP for the six neural faNN models.

Global batch = up to world_size distinct cases; no repeated or dropped samples.
Validation/test sufficient statistics are reduced, never rank RMSE averages.
UNet uses ordinary local BatchNorm with DDP rank-zero buffer broadcasts.
"""
from tfai.benchmarks.fann_logging import print, log_every, should_log

import copy
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.data import Data
from .fann_sampling import SCHEMA as SAMPLING_SCHEMA,registered_geometry_keys,geometry_batches,geometry_owners,query_counts,append_records
from .fann_timing import DistributedTiming
from .fann_geo_diagnostics import GeoQueryTrace
from .fann import FIELDS,Normalizer
from .fann_training import (FORMAT_VERSION,RunLockedError,assert_resume_config,atomic_json,
                           build_model,evaluate,file_hash,prepare_input,store_for,execution_config,GRAPH_MODELS)
from tfai.training.optimization import build_lr_scheduler
from tfai.utils.random_utils import set_global_seed
from tfai.utils.torch_utils import atomic_torch_save


SUPPORTED={'UNet','GCN','Graph_UNet','Graph_Transformer','Transolver_3','GeoTransolver'}


def mean_case_gradients(model,bucket_bytes=25*1024**2):
    """One synchronous DDP-equivalent mean after variable-count local partitions.

    Absent ranks contribute zero; globally unused parameters remain None so Adam
    does not spuriously advance their momentum. All ranks use identical buckets.
    """
    parameters=list(model.parameters());device=parameters[0].device
    used=torch.tensor([p.grad is not None for p in parameters],dtype=torch.int32,device=device)
    dist.all_reduce(used,op=dist.ReduceOp.MAX);flags=used.cpu().tolist()
    bucket=[];size=0;world=dist.get_world_size()
    def reduce_bucket():
        if not bucket:return
        flat=torch.cat([p.grad.reshape(-1) for p in bucket]);dist.all_reduce(flat);flat.div_(world)
        offset=0
        for parameter in bucket:
            count=parameter.numel();parameter.grad.copy_(flat[offset:offset+count].view_as(parameter));offset+=count
    for parameter,active in zip(parameters,flags):
        if not active:parameter.grad=None;continue
        if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
        if bucket and (parameter.dtype!=bucket[0].dtype or size+parameter.numel()*parameter.element_size()>bucket_bytes):
            reduce_bucket();bucket=[];size=0
        bucket.append(parameter);size+=parameter.numel()*parameter.element_size()
    reduce_bucket()


def local_rng(device):
    result={'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()}
    if device.type=='cuda':result['cuda_local']=torch.cuda.get_rng_state(device)
    return result


def restore_local_rng(state,device):
    if not state:return
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    if device.type=='cuda':
        if 'cuda_local' in state:torch.cuda.set_rng_state(state['cuda_local'],device)
        elif state.get('cuda'):torch.cuda.set_rng_state(state['cuda'][0],device)


def stopped(local,device,stop_file=None):
    flag=torch.tensor(int(local or (stop_file is not None and Path(stop_file).exists())),device=device)
    dist.all_reduce(flag,op=dist.ReduceOp.MAX)
    return bool(flag.item())


def sync_buffers(model):
    for buffer in model.buffers():dist.broadcast(buffer,src=0)


def dummy_input(model,device):
    if model.name=='UNet':
        return (torch.zeros(1,model.input_dim+1,model.grid,model.grid,model.grid,device=device),
                torch.zeros(1,1,1,1,3,device=device))
    return Data(x=torch.zeros(8,model.input_dim,device=device))


def distributed_evaluate(model,norm,manifest,split,device,folder=None,near_wall=.001,stop_requested=lambda:False):
    rank=dist.get_rank();world=dist.get_world_size();store=store_for(manifest,split)
    sync_buffers(model);model.eval();totals=np.zeros(34,np.float64);rows=[]
    for start in range(0,len(store),world):
        index=start+rank
        if index<len(store):
            output=None if folder is None else Path(folder)/f'rank_{rank:05d}'/f'case_{index:05d}'
            metric=evaluate(model,norm,[store[index]],device,output=output,near_wall=near_wall,return_statistics=True)
            s=metric['statistics']
            totals+=np.r_[s['sse'],s['absolute'],s['total'],s['square'],s['near_sse'],
                          s['normalized_sse'],s['count'],s['near_count'],1]
            rows.extend(metric['samples'])
        if stopped(stop_requested(),device):return None
    tensor=torch.tensor(totals,dtype=torch.float64,device=device);dist.all_reduce(tensor)
    values=tensor.cpu().numpy();sse,absolute,total,square,near_sse=np.split(values[:30],5)
    normalized_sse,count,near_count,cases=values[30:];denominator=square-total**2/count
    packets=[None]*world;dist.all_gather_object(packets,rows)
    return {'normalized_mse':float(normalized_sse/(count*6)),'nodes':int(count),'cases':int(cases),
            'fields':list(FIELDS),'units':['m/s','m/s','m/s','Pa','K','kg/m^3'],
            'rmse':np.sqrt(sse/count).tolist(),'mae':(absolute/count).tolist(),
            'r2':[float(1-sse[i]/denominator[i]) if denominator[i]>0 else None for i in range(6)],
            'near_wall':{'threshold_m':near_wall,'nodes':int(near_count),
                         'rmse':np.sqrt(near_sse/near_count).tolist() if near_count else None},
            'samples':sorted([row for packet in packets for row in packet],key=lambda row:row['source'])}


def train_ddp(name,manifest,config,folder,device,resume=False,max_updates=None,checkpoint_every=10,stop_file=None):
    rank=dist.get_rank();folder=Path(folder);lock=None;error=[None]
    if rank==0:
        try:
            folder.mkdir(parents=True,exist_ok=True);lock=(folder/'run.lock').open('a')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:error[0]='Another process holds '+str(folder)
        except Exception as exception:error[0]=str(exception)
    dist.broadcast_object_list(error,src=0)
    if error[0]:
        if lock is not None:lock.close()
        raise RunLockedError(error[0])
    try:return _train_ddp(name,manifest,config,folder,torch.device(device),resume,max_updates,checkpoint_every,stop_file)
    except Exception as exception:
        if rank==0:atomic_json({'state':'failed','model':name,'error':str(exception)},folder/'status.json')
        raise
    finally:
        if lock is not None:lock.close()


def _train_ddp(name,manifest,config,folder,device,resume,max_updates,checkpoint_every,stop_file):
    rank=dist.get_rank();world=dist.get_world_size()
    if name not in SUPPORTED:raise ValueError('Unsupported DDP model: '+name)
    if checkpoint_every<1:raise ValueError('Checkpoint interval must be positive')
    config=execution_config(name,config)
    manual=name in GRAPH_MODELS
    if manual and config.get('graph_execution')!='bounded':raise ValueError('Graph-model DDP requires --graph-execution bounded')
    last_path=folder/'last.pt';best_path=folder/'best.pt'
    if not resume and (last_path.exists() or best_path.exists()):raise ValueError('Existing run; use --resume or a new output directory')
    if resume and not last_path.exists():raise ValueError('No last.pt to resume')
    saved=torch.load(last_path,map_location='cpu',weights_only=False) if resume else None
    if saved:
        if saved.get('format_version')!=FORMAT_VERSION or saved['model']!=name:raise ValueError('Wrong checkpoint format/model')
        if saved['manifest']['signature']!=manifest['signature']:raise ValueError('Dataset/split contents changed')
        assert_resume_config(saved['config'],config)
        if saved['completed_epochs']>0 and not best_path.exists():raise ValueError('Missing validation-best checkpoint')
    norm_packet=[saved['normalization'] if saved else manifest.get('training_normalization')]
    if rank==0 and norm_packet[0] is None:norm_packet[0]=vars(Normalizer(store_for(manifest,'train')))
    dist.broadcast_object_list(norm_packet,src=0);norm=Normalizer.from_state(norm_packet[0])
    set_global_seed(config['seed'])
    model=build_model(name,[],norm,config,state=saved['state_dict'] if saved else None).to(device)
    if manual:
        for parameter in model.parameters():dist.broadcast(parameter.data,src=0)
        sync_buffers(model);ddp=None
    else:ddp=DDP(model,device_ids=[device.index] if device.type=='cuda' else None,broadcast_buffers=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    scheduler=build_lr_scheduler(optimizer,{'lr_scheduler':{'enabled':True,'factor':config['lr_factor'],
        'patience':config['lr_patience'],'threshold':config['lr_threshold'],'cooldown':config['lr_cooldown'],'min_lr':config['min_lr']}})
    epoch=position=global_step=bad=0;best=math.inf;history=[];pending_sum=0.;pending_count=0;elapsed=0.
    set_global_seed(config['seed']+rank)
    if saved:
        optimizer.load_state_dict(saved['optimizer_state_dict']);scheduler.load_state_dict(saved['scheduler_state_dict'])
        epoch=saved['completed_epochs'];position=saved['position'];global_step=saved['global_step']
        best=saved['best_score'];bad=saved['early_stopping_counter'];history=copy.deepcopy(saved['history'])
        pending_sum=saved['pending_loss_sum'];pending_count=saved['pending_elements'];elapsed=saved.get('training_seconds',0.)
        previous=saved.get('distributed_state',{})
        if previous.get('world_size')==world:restore_local_rng(previous['rng_states'][rank],device)
        else:
            restore_local_rng(saved.get('rng_state'),device)
            if rank==0:print(f'{name}: importing checkpoint with changed global batch; exact trajectory is not preserved',flush=True)
    train=store_for(manifest,'train',config['cache_samples'])
    if position<0 or position>len(train):raise ValueError('Invalid checkpoint sample position')
    affinity=config.get('sampler')=='geometry_affinity'
    geometry_keys=None;sampling_records=[]
    if affinity:
        packet=[None]
        if rank==0:
            try:
                geometry_keys=saved.get('geometry_keys') if saved else None
                if geometry_keys is None:geometry_keys=registered_geometry_keys(manifest,progress=print)
                if len(geometry_keys)!=len(train) or not all(isinstance(key,str) and key for key in geometry_keys):raise ValueError('Invalid geometry index')
                packet[0]={'keys':geometry_keys}
            except Exception as error:packet[0]={'error':str(error)}
        dist.broadcast_object_list(packet,src=0)
        if 'error' in packet[0]:raise ValueError(packet[0]['error'])
        geometry_keys=packet[0]['keys']
        if saved and saved.get('sampling_schema')!=SAMPLING_SCHEMA:raise ValueError('Sampling schema changed')
        owners=geometry_owners(geometry_keys,world,config['seed'])
        if rank==0:
            atomic_json({'schema':'geometry-affinity-v1','dataset_signature':manifest['signature'],'world_size':world,
                'preferred_owners':owners,'train_geometry_keys':geometry_keys,
                'note':'Preferred ranks with work stealing; no duplicate or dropped samples'},folder/'geometry_affinity.json')
            print(f'{name}: geometry affinity enabled; geometries={len(set(geometry_keys))}; within-geometry case order shuffled each epoch')
    interval=log_every()
    timing=DistributedTiming(folder,device)
    query_trace=GeoQueryTrace(model)
    if timing.limit and name=='GeoTransolver':query_trace.__enter__()
    stop_local=False;updates=0;started=time.perf_counter();handlers={};finalizing=False
    def request_stop(_signal,_frame):
        nonlocal stop_local
        stop_local=True
    for number in (signal.SIGTERM,signal.SIGINT):handlers[number]=signal.signal(number,request_stop)
    def wants_stop():return stop_local or (folder/'STOP').exists() or (stop_file is not None and Path(stop_file).exists())
    def status(state,**extra):
        if rank==0:atomic_json({'state':state,'model':name,'completed_epochs':epoch,'global_step':global_step,
                               'position':position,'world_size':world,'scope':manifest['scope'],**extra},folder/'status.json')
    def save(path):
        if affinity:append_records(folder,rank,sampling_records)
        timing_start=timing.begin()
        states=[None]*world;dist.all_gather_object(states,local_rng(device))
        if rank==0:
            checkpoint={'format_version':FORMAT_VERSION,'model':name,'state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                'normalization':vars(norm),'config':config,'manifest':manifest,'fields':list(FIELDS),
                **({'geometry_keys':geometry_keys,'sampling_schema':'geometry-affinity-v1'} if affinity else {}),
                'completed_epochs':epoch,'position':position,'global_step':global_step,'best_score':best,
                'early_stopping_counter':bad,'history':history,'pending_loss_sum':pending_sum,'pending_elements':pending_count,
                'rng_state':states[0],'training_seconds':elapsed+time.perf_counter()-started,
                'distributed_state':{'world_size':world,'rng_states':states,'global_batch_cases':world,
                                     'batch_norm':'local BatchNorm; rank0 buffers broadcast'},
                'execution_runtime':{'ddp_backend':dist.get_backend(),'gradient_sync':'case_allreduce' if manual else 'native_ddp',
                                     'transolver_chunk':getattr(model.net,'chunk_size',None),
                                     'geo_activation_offload':getattr(model.net,'activation_offload',False),
                                     'geo_node_chunk':getattr(getattr(model.net,'model',None),'node_chunk_size',None)}}
            atomic_torch_save(checkpoint,path)
        dist.barrier()
        phases={};timing.mark(phases,'checkpoint_total',timing_start)
        timing.emit('checkpoint',phases,step=global_step,epoch=epoch,path=str(path))
    def timed_evaluate(split,output=None):
        timing_start=timing.begin()
        result=distributed_evaluate(model,norm,manifest,split,device,output,config['near_wall'],wants_stop)
        phases={};timing.mark(phases,'evaluation_total',timing_start)
        timing.emit('evaluation',phases,step=global_step,epoch=epoch,split=split,write_predictions=output is not None)
        return result
    def interrupted():
        if not finalizing:save(last_path)
        status('interrupted')
        return {'state':'interrupted','global_step':global_step,'world_size':world}
    try:
        if rank==0:print(f'{name}: DDP world_size={world}, global_batch<= {world} full cases, rank-local BatchNorm, no LR scaling; amp={config.get("amp","off")}; geo_activation_offload={getattr(model.net,"activation_offload",False)}; geo_node_chunk={getattr(getattr(model.net,"model",None),"node_chunk_size",None)}',flush=True)
        status('training');save(last_path)
        while epoch<config['epochs'] and bad<config['patience']:
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist()
            batches=(geometry_batches(geometry_keys,world,config['seed'],epoch) if affinity else
                     [order[start:start+world]+[None]*max(0,start+world-len(order)) for start in range(0,len(order),world)])
            while position<len(order):
                if stopped(wants_stop() or (max_updates is not None and updates>=max_updates),device):return interrupted()
                model.train()
                if ddp is not None:ddp.train()
                optimizer.zero_grad(set_to_none=True);step_start=time.perf_counter()
                profiling=updates<timing.limit
                query_trace.reset(profiling)
                source=None
                phase_times={};phase_start=timing.begin(profiling);profile_start=phase_start
                end=min(position+world,len(order));batch_cases=end-position
                batch=batches[position//world];sample_index=batch[rank];active=sample_index is not None
                before_queries=query_counts(model) if affinity else None
                batch_rng=local_rng(device) if manual else None;aborted=False
                global_elements=sum(manifest['splits']['train'][index]['nodes']*6 for index in batch if index is not None)
                if active:
                    sample=train[sample_index];elements=sample['targets'].size
                    source=train.paths[sample_index]
                    phase_start=timing.mark(phase_times,'load_sample',phase_start)
                    if manual and name in ('GCN','Graph_Transformer'):
                        value=model.graph_executor.backward_sample(model,norm,sample,device,
                            stop_requested=wants_stop,loss_scale=world*elements/global_elements)
                        phase_start=timing.mark(phase_times,'input_forward_backward_combined',phase_start)
                        aborted=value is None
                        squared=torch.tensor(0. if aborted else value*elements,device=device,dtype=torch.float64)
                    else:
                        if manual:
                            prediction=model.graph_executor.graph_unet(model,norm,sample,device)
                        else:
                            inputs=prepare_input(model,norm,sample,device)
                            phase_start=timing.mark(phase_times,'prepare_input',phase_start)
                            prediction=ddp(inputs);del inputs
                        phase_start=timing.mark(phase_times,'forward',phase_start)
                        target=norm.target(sample,device)
                        if prediction.shape!=target.shape:raise ValueError('Output shape mismatch')
                        squared=(prediction-target).square().sum()
                        del target
                elif manual:
                    squared=torch.zeros((),device=device);elements=0
                else:
                    # Native DDP ranks still need one tiny, zero-gradient forward.
                    ddp.eval();prediction=ddp(dummy_input(model,device));ddp.train()
                    squared=prediction.sum()*0.;elements=0
                phase_start=timing.mark(phase_times,'target_loss_or_idle',phase_start)
                if manual and stopped(aborted or wants_stop(),device):
                    optimizer.zero_grad(set_to_none=True);restore_local_rng(batch_rng,device)
                    return interrupted()
                aggregate=torch.tensor([float(squared.detach()),elements],dtype=torch.float64,device=device)
                bad_loss=torch.tensor(int(not torch.isfinite(squared).item()),device=device);dist.all_reduce(bad_loss,op=dist.ReduceOp.MAX)
                if bad_loss.item():raise ValueError('Nonfinite DDP training loss')
                dist.all_reduce(aggregate)
                phase_start=timing.mark(phase_times,'loss_collectives',phase_start)
                query_trace.phase='backward'
                if not (manual and name in ('GCN','Graph_Transformer')):
                    loss=squared*(world/aggregate[1])
                    if active or not manual:loss.backward()
                phase_start=timing.mark(phase_times,'backward_with_ddp_sync' if not manual else 'backward',phase_start)
                if manual:mean_case_gradients(model)
                phase_start=timing.mark(phase_times,'manual_gradient_sync',phase_start)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();global_step+=1;updates+=1;position=end
                phase_start=timing.mark(phase_times,'clip_optimizer',phase_start)
                pending_sum+=float(aggregate[0]);pending_count+=int(aggregate[1])
                if not (manual and name in ('GCN','Graph_Transformer')):
                    if active or not manual:del prediction
                    del loss
                del squared
                if active:del sample
                if rank==0 and should_log(math.ceil(position/world),math.ceil(len(train)/world),interval):
                    print(f'{name} DDP epoch={epoch+1}/{config["epochs"]} batch={math.ceil(position/world)}/{math.ceil(len(train)/world)} step={global_step} epoch_cases={position}/{len(train)} batch_cases={batch_cases} loss={float(aggregate[0]/aggregate[1]):.7g} rank0_seconds={time.perf_counter()-step_start:.3f}',flush=True)
                if affinity:
                    after_queries=query_counts(model)
                    sampling_records.append({'step':global_step,'epoch':epoch+1,'rank':rank,'active':active,
                        'sample_index':sample_index,'source':source,
                        'geometry_key':geometry_keys[sample_index] if active else None,
                        'preferred_owner':owners[geometry_keys[sample_index]] if active else None,
                        'cache_delta':{key:after_queries[key]-before_queries[key] for key in after_queries}})
                if profiling:
                    phase_start=timing.mark(phase_times,'bookkeeping',phase_start)
                    phase_times['step_total']=phase_start-profile_start
                    timing.emit('train_step',phase_times,step=global_step,epoch=epoch+1,active=active,batch_cases=batch_cases,source=source,geo=query_trace.report())
                if updates%checkpoint_every==0:save(last_path);status('training')
            if stopped(wants_stop() or (max_updates is not None and updates>=max_updates),device):return interrupted()
            metrics=timed_evaluate('valid')
            if metrics is None:return interrupted()
            value=metrics['normalized_mse'];scheduler.step(value);epoch+=1;position=0
            improved=value<best
            if improved:best=value;bad=0
            else:bad+=1
            history.append({'epoch':epoch,'global_step':global_step,'train_loss':pending_sum/pending_count,
                            'val_loss':value,'lr':optimizer.param_groups[0]['lr'],'validation_rmse':metrics['rmse']})
            pending_sum=0.;pending_count=0
            if improved:save(best_path)
            save(last_path)
            if rank==0:
                atomic_json(history,folder/'history.json')
                print(f'{name} DDP epoch={epoch}/{config["epochs"]} step={global_step} train={history[-1]["train_loss"]:.7g} val={value:.7g} lr={optimizer.param_groups[0]["lr"]:.4g} bad={bad}/{config["patience"]}',flush=True)
            status('training',best_validation_loss=best,lr=optimizer.param_groups[0]['lr'])
        training_seconds=elapsed+time.perf_counter()-started;finalizing=True
        best_saved=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(best_saved['state_dict'],strict=True)
        status('evaluating',best_epoch=best_saved['completed_epochs'])
        validation=timed_evaluate('valid',folder/'predictions/valid')
        if validation is None:return interrupted()
        test=timed_evaluate('test',folder/'predictions/test')
        if test is None:return interrupted()
        report={'state':'complete','model':name,'completed_epochs':epoch,'best_epoch':best_saved['completed_epochs'],
                'optimizer_steps':global_step,'early_stopped':bad>=config['patience'],'best_checkpoint_sha256':file_hash(best_path),
                'dataset_signature':manifest['signature'],'scope':manifest['scope'],'validation':validation,'test':test,
                'training_seconds':training_seconds,'fields':list(FIELDS),'config':config,
                'selection':'minimum complete-validation normalized MSE; test only after selection',
                'environment':{'torch':torch.__version__,'device':str(device),'ddp_backend':dist.get_backend(),'world_size':world,
                               'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                               'slurm_nodes':os.environ.get('SLURM_NNODES'),'global_batch_cases':world,
                               'batch_norm':'local BatchNorm; rank0 buffers broadcast',
                               'gradient_sync':'case_allreduce' if manual else 'native_ddp'}}
        if rank==0:atomic_json(report,folder/'report.json')
        status('complete',best_epoch=best_saved['completed_epochs']);dist.barrier()
        return report
    finally:
        if affinity:append_records(folder,rank,sampling_records)
        query_trace.__exit__()
        for number,handler in handlers.items():signal.signal(number,handler)
