"""Deterministic geometry-preferred DDP batches, exact coverage and bounded imbalance."""
from collections import defaultdict,deque
import fcntl
import json
import os
from pathlib import Path
import torch

SCHEMA='geometry-affinity-v1'


def registered_geometry_keys(manifest,progress=print):
    from .fann_precompute import geometry_index
    home=Path(os.environ.get('TURBINE_FANN_CACHE_WORK_ROOT','runs/fann_cache_work_v1'))
    home.mkdir(parents=True,exist_ok=True)
    with (home/('geometry_'+manifest['signature']+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index=geometry_index(manifest,home,progress)
    return [index['sources'][row['path']] for row in manifest['splits']['train']]


def geometry_owners(keys,world,seed):
    if world<1:raise ValueError('world must be positive')
    groups=defaultdict(list)
    for index,key in enumerate(keys):groups[key].append(index)
    # Seeded ties, largest groups first; every geometry gets a persistent preferred rank.
    generator=torch.Generator().manual_seed(seed)
    names=sorted(groups)
    names=[names[i] for i in torch.randperm(len(names),generator=generator).tolist()]
    names.sort(key=lambda key:len(groups[key]),reverse=True)
    counts=[0]*world;owners={}
    for key in names:
        rank=min(range(world),key=lambda rank:(counts[rank],rank))
        owners[key]=rank;counts[rank]+=len(groups[key])
    return owners


def geometry_batches(keys,world,seed,epoch):
    owners=geometry_owners(keys,world,seed);groups=defaultdict(list)
    for index,key in enumerate(keys):groups[key].append(index)
    generator=torch.Generator().manual_seed(seed+epoch)
    queues=[]
    for rank in range(world):
        names=sorted(key for key in groups if owners[key]==rank)
        names=[names[i] for i in torch.randperm(len(names),generator=generator).tolist()]
        queue=[]
        for key in names:
            members=groups[key]
            queue.extend(members[i] for i in torch.randperm(len(members),generator=generator).tolist())
        queues.append(deque(queue))
    remaining=len(keys);batches=[]
    while remaining:
        batch=[None]*world
        # Serve local queues before stealing, so an empty rank cannot steal the
        # only available sample from a rank which has not been served yet.
        for rank in range(world):
            if queues[rank]:batch[rank]=queues[rank].popleft();remaining-=1
        for rank in range(world):
            if batch[rank] is None and remaining:
                donor=max(range(world),key=lambda rank:len(queues[rank]))
                batch[rank]=queues[donor].popleft();remaining-=1
        batches.append(batch)
    return batches


def query_counts(model):
    from tfai.models._vendor.geotransolver2.ball_query import BQWarp
    names=('device_cache_hits','memory_cache_hits','disk_cache_hits','tree_builds')
    return {key:sum(getattr(module,key) for module in model.modules() if isinstance(module,BQWarp)) for key in names}


def append_records(folder,rank,records):
    if not records:return
    directory=Path(folder)/'sampling';directory.mkdir(parents=True,exist_ok=True)
    with (directory/f'rank_{rank:05d}.jsonl').open('a') as stream:
        for row in records:stream.write(json.dumps(row)+'\n')
    records.clear()
