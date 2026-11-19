"""CPU-only faNN format definitions; no model or VTK imports."""
import pickle

FIELDS = ("VelocityX", "VelocityY", "VelocityZ", "Pressure", "Temperature", "Density")

class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'),
                   ('numpy', 'dtype'), ('numpy.core.numeric', '_frombuffer')}
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')
        return super().find_class(module, name)



def select_structured_patches(connectivity, indices, blocks, count, seed=42):
    """Choose common compact index-space patches across blocks and three span bands.

    Connectivity is 1-based. Selection uses only shared mesh indices, never targets.
    This is a distributed set of patches, not a connected full-flow domain.
    """
    import numpy as np
    if not 1 <= count <= len(connectivity): raise ValueError('Invalid cell count')
    corners = connectivity[:,0].astype(np.int64)-1
    block_ids = np.asarray(blocks)[corners].astype(np.int32)
    ijk = np.column_stack([np.asarray(axis)[corners] for axis in indices]).astype(np.float32)
    groups=[]
    for block in np.unique(block_ids):
        ids=np.flatnonzero(block_ids==block)
        span=ijk[ids,1]; lo=span.min(); width=max(float(span.max()-lo),1.)
        bands=np.minimum(((span-lo)/width*3).astype(int),2)
        for band in range(3):
            group=ids[bands==band]
            if len(group):groups.append(group)
    quota=np.zeros(len(groups),dtype=int)
    remaining=count
    while remaining:
        eligible=[i for i,g in enumerate(groups) if quota[i]<len(g)]
        portion=max(1,remaining//len(eligible))
        for i in eligible:
            take=min(portion,len(groups[i])-quota[i],remaining)
            quota[i]+=take;remaining-=take
            if not remaining:break
    rng=np.random.default_rng(seed);chosen=[]
    for group,q in zip(groups,quota):
        if not q:continue
        center=ijk[group[rng.integers(len(group))]]
        distances=np.max(np.abs(ijk[group]-center),axis=1)
        cutoff=np.partition(distances,q-1)[q-1]
        inside=group[distances<cutoff];ties=group[distances==cutoff]
        chosen.append(np.r_[inside,ties[:q-len(inside)]])
    result=np.sort(np.concatenate(chosen)).astype(np.int64)
    if len(result)!=count or len(np.unique(result))!=count:raise ValueError('Invalid patch selection')
    return result
