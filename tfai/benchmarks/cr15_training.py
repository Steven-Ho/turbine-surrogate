"""Resumable whole-case CR15 training, single GPU or DDP with accumulation.

Global case batch is independent of GPU count. Losses use actual node counts.
Validation selects the checkpoint; test and independent baselines run only at
final completion. No fabricated wall-distance/near-wall metrics.
"""
from contextlib import nullcontext
import fcntl
import json
import math
from pathlib import Path
import resource
import signal
import socket
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .cr15 import FIELDS,UNITS,CR15Normalizer,CR15Store
from .cr15_models import make_model,model_input
from .fann_ddp import local_rng,restore_local_rng
from .fann_logging import print
from .fann_training import atomic_json,RunLockedError
from tfai.utils.random_utils import set_global_seed
from tfai.utils.torch_utils import atomic_torch_save


def topology():return (dist.get_rank(),dist.get_world_size()) if dist.is_initialized() else (0,1)
def reduce_sum(value):
    if dist.is_initialized():dist.all_reduce(value)
    return value

def gather(value):
    if not dist.is_initialized():return [value]
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values

def barrier():
    if dist.is_initialized():dist.barrier()

def synchronize(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def evaluate(model,norm,prepared,split,device,output=None,stop=lambda:False):
    rank,world=topology();store=CR15Store(prepared['paths'][split]);channels=len(FIELDS)
    if not len(store):return None
    model.eval();totals=np.zeros(channels*4+3,dtype='float64');rows=[]
    with torch.inference_mode():
        for start in range(0,len(store),world):
            if stop():return None
            index=start+rank
            if index<len(store):
                sample=store[index];prediction=model(model_input(model,norm,sample,device))
                if prediction.shape!=sample['targets'].shape or not torch.isfinite(prediction).all():raise ValueError('Invalid prediction')
                normalized_sse=float((prediction-norm.target(sample,device)).double().square().sum())
                physical=norm.decode(prediction);truth=sample['targets'];n=len(truth)
                # Bounded float64 temporaries; no all-case stacking.
                sse=np.zeros(channels);absolute=np.zeros(channels);total=np.zeros(channels);square=np.zeros(channels)
                for begin in range(0,n,65536):
                    a=truth[begin:begin+65536].astype('float64');error=physical[begin:begin+65536].astype('float64')-a
                    sse+=(error**2).sum(0);absolute+=np.abs(error).sum(0);total+=a.sum(0);square+=(a**2).sum(0)
                totals+=np.r_[sse,absolute,total,square,normalized_sse,n,1]
                row=dict(source=sample['source'],sample=Path(sample['source']).stem,nodes=n,
                         rmse=np.sqrt(sse/n).tolist(),mae=(absolute/n).tolist())
                if output is not None:
                    folder=Path(output)/f'rank_{rank:05d}'/f'case_{index:05d}';folder.mkdir(parents=True,exist_ok=True)
                    path=folder/'prediction_00000.npz';temporary=folder/'prediction_00000.tmp.npz'
                    np.savez_compressed(temporary,prediction=physical,original_node_ids=sample['original_node_ids'],
                                        output_names=np.array(FIELDS),source=np.array(sample['source']))
                    temporary.replace(path);row['prediction_file']=str(path.resolve())
                rows.append(row)
    values=reduce_sum(torch.tensor(totals,device=device)).cpu().numpy()
    sse,absolute,total,square=np.split(values[:channels*4],4);normalized_sse,count,cases=values[-3:]
    denominator=square-total**2/count
    all_rows=[r for packet in gather(rows) for r in packet]
    return dict(fields=FIELDS,units=UNITS,nodes=int(count),cases=int(cases),
        normalized_mse=float(normalized_sse/(count*channels)),rmse=np.sqrt(sse/count).tolist(),
        mae=(absolute/count).tolist(),r2=[float(1-a/b) if b>0 else None for a,b in zip(sse,denominator)],
        samples=sorted(all_rows,key=lambda row:row['source']))


def train(prepared,config,folder,device,resume=False,epoch_budget=0,max_updates=None,stop_file=None):
    rank,world=topology();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    lock=None;error=None
    if rank==0:
        try:
            lock=(folder/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:error='Another process holds '+str(folder)
        except Exception as exception:error=str(exception)
    errors=gather(error)
    if any(errors):
        if lock:lock.close()
        raise RunLockedError(str(errors))
    try:return _train(prepared,config,folder,device,resume,epoch_budget,max_updates,stop_file)
    finally:
        if lock:lock.close()


def _train(prepared,config,folder,device,resume,epoch_budget,max_updates,stop_file):
    rank,world=topology();last=folder/'last.pt';best_path=folder/'best.pt'
    name=config.get('model','GeoTransolver')
    if world>config['batch_cases']:raise ValueError('GPU count exceeds configured global batch_cases')
    if not resume and (last.exists() or best_path.exists()):raise ValueError('Existing experiment: use --resume or a new output')
    saved=torch.load(last,map_location='cpu',weights_only=False) if resume else None
    if saved:
        if saved.get('schema')!='cr15-training-v1' or saved['cache_identity']!=prepared['identity']:
            raise ValueError('Checkpoint/cache identity differs')
        for key,value in config.items():
            if key not in ['epochs','threads'] and saved['config'].get(key)!=value:
                raise ValueError('Resume configuration changed: '+key)
        if saved['completed_epochs']>config['epochs']:raise ValueError('Target epochs precede checkpoint')
    norm=CR15Normalizer(prepared)
    if saved and saved['normalization']!=norm.state():raise ValueError('Train-only normalization differs')
    if rank==0:atomic_json(dict(config=config,cache_identity=prepared['identity'],prepared_root=prepared['root'],
        normalization=norm.state(),fields=FIELDS,units=UNITS,world_size=world,
        input_names=['x','y','z']+prepared['contract']['boundary_names']),folder/'configuration.json')
    set_global_seed(config['seed'])
    model=make_model(prepared,config,saved).to(device)
    if saved:model.load_state_dict(saved['state_dict'],strict=True)
    ddp=DDP(model,device_ids=[device.index] if device.type=='cuda' else None) if world>1 else None
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],betas=(config['beta1'],config['beta2']))
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=config['lr_factor'],
        patience=config['lr_patience'],threshold=config['lr_threshold'],threshold_mode='rel',
        cooldown=config['lr_cooldown'],min_lr=config['min_lr'])
    epoch=position=step=bad=0;best=math.inf;history=[];pending_sum=0.;pending_elements=0;elapsed=0.
    set_global_seed(config['seed']+rank)
    if saved:
        optimizer.load_state_dict(saved['optimizer_state_dict']);scheduler.load_state_dict(saved['scheduler_state_dict'])
        epoch=saved['completed_epochs'];position=saved['position'];step=saved['global_step']
        bad=saved['early_stopping_counter'];best=saved['best_score'];history=saved['history']
        pending_sum=saved['pending_sum'];pending_elements=saved['pending_elements'];elapsed=saved['training_seconds']
        if saved['world_size']==world:restore_local_rng(saved['rng_states'][rank],device)
        elif rank==0:print('CR15 resume: GPU count changed; global case batch retained, exact numerical trajectory not guaranteed',flush=True)
    if epoch and not best_path.is_file():raise ValueError('Missing best validation checkpoint')
    stores={split:CR15Store(paths) for split,paths in prepared['paths'].items()}
    initial_epoch=epoch;updates=0;started=time.perf_counter();stop_local=False;finalizing=False
    previous={}
    def handler(_number,_frame):
        nonlocal stop_local
        stop_local=True
    for number in [signal.SIGTERM,signal.SIGINT]:previous[number]=signal.signal(number,handler)
    def wants_stop():
        local=stop_local or (folder/'STOP').exists() or (stop_file is not None and Path(stop_file).exists())
        flag=torch.tensor(int(local),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        return bool(flag.item())
    def capacity():
        return dict(rank=rank,host=socket.gethostname(),device=str(device),torch=torch.__version__,gpu_peak_GiB=torch.cuda.max_memory_allocated(device)/1024**3 if device.type=='cuda' else 0.,
                    process_peak_RSS_GiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    def status(state,**extra):
        if rank==0:atomic_json(dict(state=state,model=name,dataset='CR15',completed_epochs=epoch,
            global_step=step,position=position,world_size=world,global_batch_cases=config['batch_cases'],
            best_validation_loss=best if math.isfinite(best) else None,**extra),folder/'status.json')
    def save(path):
        states=gather(local_rng(device))
        if rank==0:
            atomic_torch_save(dict(schema='cr15-training-v1',model=name,fields=FIELDS,
                cache_identity=prepared['identity'],config=config,normalization=norm.state(),
                state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
                completed_epochs=epoch,position=position,global_step=step,best_score=best,early_stopping_counter=bad,
                pending_sum=pending_sum,pending_elements=pending_elements,history=history,
                training_seconds=elapsed+time.perf_counter()-started,rng_states=states,world_size=world),path)
        barrier()
    def interrupted(reason):
        if not finalizing:save(last)
        stats=gather(capacity());status('interrupted',stop_reason=reason,capacity=stats)
        if rank==0:print(f'CR15 planned/safe stop: reason={reason}, epoch={epoch}, step={step}, capacity={stats}',flush=True)
        return dict(state='interrupted',stop_reason=reason,global_step=step)
    try:
        if rank==0:
            print(f'CR15 {name} scale={config["scale"]} fields=5 global_conditions=6 world_size={world} global_batch<={config["batch_cases"]} full cases; gradient accumulation, node-weighted loss; amp={config["amp"]}',flush=True)
            print(f'CR15 cache reused: train={len(stores["train"])} valid={len(stores["valid"])} test={len(stores["test"])} baseline={len(stores["baseline"])}; train-only statistics; no raw-file scan',flush=True)
            print(f'Adam lr={optimizer.param_groups[0]["lr"]} betas={optimizer.param_groups[0]["betas"]}; LR patience={config["lr_patience"]} threshold={config["lr_threshold"]}',flush=True)
        status('training');save(last)
        while epoch<config['epochs'] and bad<config['patience']:
            order=(sorted(range(len(stores['train'])),key=lambda i:(-prepared['nodes']['train'][i],i)) if config['largest_first'] else
                   torch.randperm(len(stores['train']),generator=torch.Generator().manual_seed(config['seed']+epoch)).tolist())
            while position<len(order):
                if wants_stop():return interrupted('signal_or_stop_file')
                if max_updates is not None and updates>=max_updates:return interrupted('update_budget')
                model.train()
                if ddp:ddp.train()
                optimizer.zero_grad(set_to_none=True);synchronize(device);batch_started=time.perf_counter()
                selected=order[position:position+config['batch_cases']]
                elements=sum(prepared['nodes']['train'][i]*len(FIELDS) for i in selected)
                total_squared=torch.zeros((),device=device,dtype=torch.float64)
                micros=math.ceil(len(selected)/world)
                for micro in range(micros):
                    slot=micro*world+rank
                    context=ddp.no_sync() if ddp is not None and micro<micros-1 else nullcontext()
                    with context:
                        if slot<len(selected):
                            sample=stores['train'][selected[slot]]
                            prediction=(ddp or model)(model_input(model,norm,sample,device))
                            target=norm.target(sample,device)
                            squared=(prediction-target).square().sum()
                        else:
                            dummy=stores['train'][order[0]]
                            prediction=(ddp or model)(model_input(model,norm,dummy,device))
                            squared=prediction.sum()*0.
                        # DDP divides gradients by world; undo that once, including tail batches.
                        (squared*(world/elements)).backward()
                        total_squared+=squared.detach().double()
                reduce_sum(total_squared)
                if not torch.isfinite(total_squared):raise ValueError('Nonfinite training loss')
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['grad_clip'],error_if_nonfinite=True)
                optimizer.step();synchronize(device)
                step+=1;updates+=1;position+=len(selected)
                pending_sum+=float(total_squared);pending_elements+=elements
                if rank==0 and (step%config['log_every']==0 or position==len(order)):
                    print(f'CR15 {name} epoch={epoch+1}/{config["epochs"]} batch={math.ceil(position/config["batch_cases"])}/{math.ceil(len(order)/config["batch_cases"])} step={step} cases={position}/{len(order)} batch_cases={len(selected)} loss={float(total_squared)/elements:.7g} lr={optimizer.param_groups[0]["lr"]:.4g} rank0_seconds={time.perf_counter()-batch_started:.3f} peak_GiB={capacity()["gpu_peak_GiB"]:.3f}',flush=True)
                if updates%config['checkpoint_every']==0:save(last);status('training')
            if max_updates is not None and updates>=max_updates:return interrupted('update_budget')
            valid=evaluate(model,norm,prepared,'valid',device,stop=wants_stop)
            if valid is None:return interrupted('signal_or_stop_file')
            value=valid['normalized_mse'];scheduler.step(value);epoch+=1;position=0
            improved=value<best
            if improved:best=value;bad=0
            else:bad+=1
            history.append(dict(epoch=epoch,global_step=step,train_loss=pending_sum/pending_elements,
                val_loss=value,lr=optimizer.param_groups[0]['lr'],validation_rmse=valid['rmse']))
            pending_sum=0.;pending_elements=0
            if improved:save(best_path)
            save(last)
            if rank==0:
                atomic_json(history,folder/'history.json')
                print(f'CR15 {name} epoch={epoch}/{config["epochs"]} step={step} train={history[-1]["train_loss"]:.7g} val={value:.7g} lr={optimizer.param_groups[0]["lr"]:.4g} bad={bad}/{config["patience"]}',flush=True)
            status('training')
            if epoch_budget and epoch-initial_epoch>=epoch_budget:return interrupted('epoch_budget')
        training_seconds=elapsed+time.perf_counter()-started;finalizing=True
        checkpoint=torch.load(best_path,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state_dict'])
        if rank==0:print(f'CR15 final evaluation: frozen best epoch={checkpoint["completed_epochs"]}; validation/test/baseline',flush=True)
        metrics={}
        for split in ['valid','test','baseline']:
            if not len(stores[split]):continue
            result=evaluate(model,norm,prepared,split,device,folder/'predictions'/split,stop=wants_stop)
            if result is None:return interrupted('signal_or_stop_file')
            metrics['validation' if split=='valid' else split]=result
        stats=gather(capacity())
        report=dict(state='complete',model=name,dataset='CR15',completed_epochs=epoch,
            optimizer_steps=step,best_epoch=checkpoint['completed_epochs'],fields=FIELDS,units=UNITS,config=config,
            training_seconds=training_seconds,early_stopped=bad>=config['patience'],capacity=stats,
            cache_identity=prepared['identity'],selection='Minimum complete-validation normalized MSE; test/baseline only after selection',
            label_quality=prepared['contract']['label_quality'],near_wall_metrics_available=False,**metrics)
        if rank==0:
            atomic_json(report,folder/'report.json')
            print(f'CR15 complete: test normalized MSE={report["test"]["normalized_mse"]:.7g}; test RMSE={report["test"]["rmse"]}; capacity={stats}',flush=True)
        status('complete',best_epoch=checkpoint['completed_epochs'],capacity=stats);barrier();return report
    except Exception as error:
        status('failed',error=str(error));raise
    finally:
        for number,old in previous.items():signal.signal(number,old)
