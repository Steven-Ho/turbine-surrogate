"""Scoped Geo radius-query tracing; never enabled for ordinary training."""
import resource
import time
import torch
from tfai.models._vendor.geotransolver2.ball_query import BQWarp

COUNTERS=('device_cache_hits','memory_cache_hits','disk_cache_hits','tree_builds','reference_chunks')


def memory_stats():
    result={'host_peak_RSS_GiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20}
    try:
        with open('/proc/self/status') as stream:
            for line in stream:
                if line.startswith('VmRSS:'):result['host_RSS_GiB']=int(line.split()[1])/2**20
    except OSError:pass
    return result


class GeoQueryTrace:
    def __init__(self,model):
        self.modules=[(name,module) for name,module in model.named_modules() if isinstance(module,BQWarp)]
        self.events=[];self.phase='forward';self.enabled=False;self.originals=[];self.cache_originals=[];self.cache_events=[]
    def sync(self,points):
        if points.device.type=='cuda':torch.cuda.synchronize(points.device)
    def __enter__(self):
        seen=set()
        for name,module in self.modules:
            cache=module.index_cache
            if id(cache) not in seen:
                seen.add(id(cache));get=cache.get
                def traced_get(*args,_get=get,**kwargs):
                    if not self.enabled:return _get(*args,**kwargs)
                    start=time.perf_counter();value,source=_get(*args,**kwargs)
                    self.cache_events.append({'seconds':time.perf_counter()-start,'source':source if value is not None else 'miss','key':args[0]})
                    return value,source
                self.cache_originals.append((cache,get));cache.get=traced_get
            original=module._indices
            def wrapped(points,queries,_original=original,_module=module,_name=name):
                if not self.enabled:return _original(points,queries)
                before={key:getattr(_module,key) for key in COUNTERS}
                cache_start=len(self.cache_events)
                self.sync(points);start=time.perf_counter()
                result=_original(points,queries)
                self.sync(points)
                delta={key:getattr(_module,key)-before[key] for key in COUNTERS}
                self.events.append({'module':_name,'phase':self.phase,'seconds':time.perf_counter()-start,
                    'radius':_module.radius,'neighbors':_module.neighbors_in_radius,
                    'points':len(points),'queries':len(queries),'cache_key':getattr(_module,'last_cache_key',None),
                    'backend':_module.backend,'cache_access':self.cache_events[cache_start:],
                    'build_and_store_seconds':_module.last_build_seconds if delta['tree_builds'] else 0.,**delta})
                return result
            self.originals.append((module,original));module._indices=wrapped
        return self
    def reset(self,enabled):
        self.enabled=enabled;self.phase='forward';self.events=[];self.cache_events=[]
    def report(self):
        return {'query_seconds_by_phase':{phase:sum(e['seconds'] for e in self.events if e['phase']==phase)
                     for phase in ('forward','backward')},'queries':list(self.events),**memory_stats()}
    def __exit__(self,*args):
        for module,original in self.originals:module._indices=original
        for cache,get in self.cache_originals:cache.get=get
        self.originals=[];self.cache_originals=[]


def summarize_probe(output):
    import csv,json
    from pathlib import Path
    output=Path(output)
    columns=['swap','iteration','rank','host','label','source','loss','load_seconds','forward_seconds',
        'backward_seconds','forward_query_seconds','cache_get_seconds','build_and_store_seconds',
        'device_hits','memory_hits','disk_hits','builds','peak_allocated_GiB','host_RSS_GiB']
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for line in (output/'timings.jsonl').read_text().splitlines():
            event=json.loads(line)
            for row in event['ranks']:
                queries=row['geo']['queries'];seconds=row['seconds']
                writer.writerow({**{key:row.get(key,'') for key in columns[:7]},
                    'load_seconds':seconds['load_sample'],'forward_seconds':seconds['forward'],
                    'backward_seconds':seconds['backward_with_ddp_sync'],
                    'forward_query_seconds':row['geo']['query_seconds_by_phase']['forward'],
                    'cache_get_seconds':sum(access['seconds'] for q in queries for access in q['cache_access']),
                    'build_and_store_seconds':sum(q['build_and_store_seconds'] for q in queries),
                    'device_hits':sum(q['device_cache_hits'] for q in queries),
                    'memory_hits':sum(q['memory_cache_hits'] for q in queries),
                    'disk_hits':sum(q['disk_cache_hits'] for q in queries),'builds':sum(q['tree_builds'] for q in queries),
                    'peak_allocated_GiB':row.get('peak_allocated_GiB',''),'host_RSS_GiB':row['geo'].get('host_RSS_GiB','')})
