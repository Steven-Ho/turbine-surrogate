"""Dataset-free FP32 AllReduce latency/bandwidth probe, one worker per GPU."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
from datetime import timedelta
from scripts.fann_diagnostic_bootstrap import load_diagnostic_helpers
_comm,print=load_diagnostic_helpers()
setup_debug,snapshot,configure_interop=_comm.setup_debug,_comm.snapshot,_comm.configure_interop
# Some vendor libraries read debugging variables at import time.
communication_root=setup_debug()
import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--backend',default='nccl')
    parser.add_argument('--sizes',type=int,nargs='+',default=[4,16,4096,1048576,8388608,33554432])
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--iterations',type=int,default=20)
    args=parser.parse_args()
    if args.warmup<1 or args.iterations<1 or any(size<4 or size%4 for size in args.sizes):parser.error('Counts positive, sizes positive multiples of 4 bytes')
    torch.set_num_threads(4);configure_interop()
    local=int(os.environ.get('LOCAL_RANK','0'))
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable')
        torch.cuda.set_device(local);device=torch.device('cuda',local)
    else:device=torch.device('cpu')
    dist.init_process_group(args.backend,timeout=timedelta(minutes=10))
    rank=dist.get_rank();world=dist.get_world_size()
    def sync():
        if device.type=='cuda':torch.cuda.synchronize(device)
    report={'state':'running','world_size':world,'backend':dist.get_backend(),'device':str(device),
            'warmup':args.warmup,'iterations':args.iterations,'results':[],
            'note':'Per-iteration fill and barrier excluded; elapsed includes AllReduce completion. Bus bandwidth is ring-normalized, not measured NIC bandwidth.'}
    try:
        if rank==0:
            args.output.mkdir(parents=True,exist_ok=True)
            if (args.output/'report.json').exists():raise ValueError('Existing AllReduce report; use a new output directory')
            (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.barrier();snapshot(communication_root,device,'initialized')
        for size in args.sizes:
            tensor=torch.empty(size//4,device=device,dtype=torch.float32);times=[]
            for iteration in range(args.warmup+args.iterations):
                tensor.fill_(1);sync();dist.barrier();sync()
                start=time.perf_counter();dist.all_reduce(tensor);sync();elapsed=time.perf_counter()-start
                if iteration>=args.warmup:times.append(elapsed)
            if not torch.all(tensor==world).item():raise ValueError('AllReduce correctness failed')
            packets=[None]*world;dist.all_gather_object(packets,{'rank':rank,'seconds':times})
            if rank==0:
                maxima=[max(p['seconds'][i] for p in packets) for i in range(args.iterations)]
                average=statistics.mean(maxima);ordered=sorted(maxima)
                row={'bytes':size,'mean_ms':average*1000,'median_ms':statistics.median(maxima)*1000,
                     'p95_ms':ordered[min(len(ordered)-1,int(.95*len(ordered)))]*1000,
                     'algorithm_GBps':size/average/1e9,'ring_normalized_bus_GBps':size/average/1e9*2*(world-1)/world,
                     'rank_timings':packets}
                report['results'].append(row)
                (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print('ALLREDUCE '+json.dumps({k:v for k,v in row.items() if k!='rank_timings'}))
            del tensor
        snapshot(communication_root,device,'completed')
        if rank==0:
            report['state']='complete';(args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
