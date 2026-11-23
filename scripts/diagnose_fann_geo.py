"""Fixed-weight A/A/B/B/A geometry and node-swap probe; run with torchrun."""
import hashlib
import json
import os
from pathlib import Path
import socket
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tfai.benchmarks.fann import Normalizer
from tfai.benchmarks.fann_training import build_model,execution_config,prepare_input,store_for
from tfai.benchmarks.fann_startup_cache import load_startup
from tfai.benchmarks.fann_geo_diagnostics import GeoQueryTrace,summarize_probe
from tfai.benchmarks.fann_timing import DistributedTiming
from tfai.benchmarks.fann_logging import print


def sequence(node,swap):
    order=['A','A','B','B','A']
    return [('B' if label=='A' else 'A') if (node+swap)%2 else label for label in order]


def main():
    from scripts.train_fann import build_parser
    parser=build_parser()
    parser.set_defaults(models=['GeoTransolver'],amp='bf16')
    parser.add_argument('--case-a',type=int,default=0)
    parser.add_argument('--case-b',type=int)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--offload',choices=['on','off'],default='on')
    parser.add_argument('--swaps',type=int,choices=[1,2],default=2)
    parser.add_argument('--backend',default='nccl')
    args=parser.parse_args()
    local=int(os.environ.get('LOCAL_RANK',0));torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU fallback')
        torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:device=torch.device('cpu')
    dist.init_process_group(args.backend);rank=dist.get_rank()
    try:
        packet=[None]
        if rank==0:
            try:
                if args.checkpoint:
                    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
                    if saved.get('model')!='GeoTransolver':raise ValueError('Expected GeoTransolver checkpoint')
                    manifest=saved['manifest'];cfg=saved['config']
                    norm=Normalizer.from_state(saved['normalization'])
                else:
                    manifest=load_startup(args.splits,progress=print)
                    norm=Normalizer.from_state(manifest['training_normalization'])
                    cfg={k:getattr(args,k) for k in ('seed','scale','grid','rank','pod_geometry_bins','streaming_pod','cache_samples')}
                # Select genuinely distinct coordinate geometries, not different conditions only.
                store=store_for(manifest,'train');a=args.case_a
                def geometry(index):
                    xyz=store[index]['coordinates'];h=hashlib.sha256()
                    h.update(str((xyz.shape,xyz.dtype.str)).encode());h.update(memoryview(xyz).cast('B'))
                    return h.hexdigest()
                ga=geometry(a);b=args.case_b
                if b is None:
                    for candidate in range(len(store)):
                        if candidate!=a and geometry(candidate)!=ga:b=candidate;break
                if b is None or geometry(b)==ga:raise ValueError('Choose two cases with different coordinates')
                packet[0]={'manifest':manifest,'config':cfg,'normalization':vars(norm),'cases':{'A':a,'B':b},'geometry_keys':{'A':ga,'B':geometry(b)}}
            except Exception as error:packet[0]={'error':str(error)}
        dist.broadcast_object_list(packet,src=0)
        if 'error' in packet[0]:raise ValueError(packet[0]['error'])
        data=packet[0];manifest=data['manifest'];cfg=dict(data['config']);cfg['amp']=args.amp
        cfg=execution_config('GeoTransolver',cfg)
        norm=Normalizer.from_state(data['normalization'])
        os.environ['TURBINE_GEO_ACTIVATION_OFFLOAD']='1' if args.offload=='on' else '0'
        torch.manual_seed(cfg['seed'])
        state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)['state_dict'] if args.checkpoint else None
        model=build_model('GeoTransolver',[],norm,cfg,state=state).to(device)
        del state
        ddp=DDP(model,device_ids=[local] if device.type=='cuda' else None)
        store=store_for(manifest,'train',cache_size=1)
        output=args.output
        if rank==0:
            output.mkdir(parents=True,exist_ok=False)
            (output/'protocol.json').write_text(json.dumps({'cases':{key:store.paths[value] for key,value in data['cases'].items()},
                'geometry_keys':data['geometry_keys'],'config':cfg,'offload':args.offload,'world_size':dist.get_world_size(),'fixed_weights':True,'optimizer_updates':0,
                'node_chunk':os.environ.get('TURBINE_GEO_NODE_CHUNK','32768')},indent=2))
        dist.barrier()
        os.environ['FANN_PROFILE_STEPS']='100'
        timing=DistributedTiming(output,device)
        def parameter_digest():
            digest=hashlib.sha256()
            for name,param in model.named_parameters():
                digest.update(name.encode());digest.update(param.detach().cpu().contiguous().numpy().tobytes())
            return digest.hexdigest()
        initial_digest=parameter_digest()
        node=int(os.environ.get('GROUP_RANK','0'))
        trace=GeoQueryTrace(model)
        if rank==0:print(f'Fixed-weight Geo probe: amp={args.amp} offload={args.offload}; no optimizer updates')
        with trace:
            for swap in range(args.swaps):
                for iteration,label in enumerate(sequence(node,swap),1):
                    # Flush only the SampleStore LRU; keep OS, geometry and device caches natural.
                    store.cache.clear();ddp.train();model.zero_grad(set_to_none=True);trace.reset(True)
                    dist.barrier()
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                    start=timing.begin();step_start=start;phases={}
                    index=data['cases'][label];sample=store[index]
                    start=timing.mark(phases,'load_sample',start)
                    inputs=prepare_input(model,norm,sample,device)
                    start=timing.mark(phases,'prepare_input',start)
                    prediction=ddp(inputs)
                    start=timing.mark(phases,'forward',start)
                    target=norm.target(sample,device);loss=(prediction-target).square().mean()
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    start=timing.mark(phases,'loss',start)
                    trace.phase='backward';loss.backward()
                    start=timing.mark(phases,'backward_with_ddp_sync',start)
                    phases['step_total']=start-step_start
                    timing.emit('fixed_case',phases,swap=swap,iteration=iteration,label=label,
                        source=store.paths[index],geometry_key=data['geometry_keys'][label],nodes=len(sample['coordinates']),loss=float(loss.detach()),
                        amp=args.amp,offload=args.offload,geo=trace.report())
                    del sample,inputs,prediction,target,loss
        final_digest=parameter_digest()
        if initial_digest!=final_digest:raise RuntimeError('Diagnostic changed model parameters')
        if rank==0:
            summarize_probe(output)
            (output/'completion.json').write_text(json.dumps({'state':'complete','parameters_unchanged':True,
                'initial_parameter_sha256':initial_digest,'final_parameter_sha256':final_digest},indent=2))
            print('Fixed-weight diagnostic complete: '+str(output))
    except Exception as error:
        if args.output.is_dir():
            (args.output/f'failure_rank_{rank}.json').write_text(json.dumps({'rank':rank,'host':socket.gethostname(),
                'offload':args.offload,'error':str(error)},indent=2))
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
