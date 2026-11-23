"""Opt-in synchronized wall-clock timings, reported by rank zero."""
import json
import os
import socket
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
import torch.distributed as dist
from .fann_logging import print


class DistributedTiming:
    def __init__(self,folder,device):
        self.limit=int(os.environ.get('FANN_PROFILE_STEPS','0'))
        if self.limit<0:raise ValueError('FANN_PROFILE_STEPS must be nonnegative')
        self.folder=Path(folder);self.device=device
    def sync(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    def begin(self,enabled=True):
        if not self.limit or not enabled:return None
        self.sync();return time.perf_counter()
    def mark(self,timer,name,start):
        if start is None:return None
        self.sync();now=time.perf_counter();timer[name]=now-start;return now
    def emit(self,event,seconds,**metadata):
        if not self.limit:return
        row={'rank':dist.get_rank(),'host':socket.gethostname(),'seconds':seconds,**metadata}
        if self.device.type=='cuda':
            row['allocated_GiB']=torch.cuda.memory_allocated(self.device)/2**30
            row['peak_allocated_GiB']=torch.cuda.max_memory_allocated(self.device)/2**30
        rows=[None]*dist.get_world_size();dist.all_gather_object(rows,row)
        if dist.get_rank()!=0:return
        phases=sorted({key for rank in rows for key in rank['seconds']})
        summary={}
        for phase in phases:
            values=[(rank['seconds'][phase],rank['rank']) for rank in rows if phase in rank['seconds']]
            slow=max(values)
            summary[phase]={'min':min(v[0] for v in values),'mean':sum(v[0] for v in values)/len(values),'max':slow[0],'max_rank':slow[1]}
        record={'time':datetime.now(timezone.utc).isoformat(),'event':event,'summary':summary,'ranks':rows}
        with (self.folder/'timings.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        print('DDP timing '+json.dumps({'event':event,**metadata,'summary':summary}))
