"""CPU spatial candidates plus content-addressed, atomic neighbor-index cache."""
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = 'geotransolver-kdtree-bounded-reference-v2'


def geometry_key(points, queries, radius, neighbors, context):
    metadata = {'schema': SCHEMA, 'radius': float(radius).hex(), 'neighbors': int(neighbors),
                'points_shape': points.shape, 'queries_shape': queries.shape,
                'points_dtype': points.dtype.str, 'queries_dtype': queries.dtype.str,
                'context': context}
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    h.update(np.ascontiguousarray(points).tobytes())
    h.update(np.ascontiguousarray(queries).tobytes())
    return h.hexdigest()


def tree_candidates(points, queries, radius, neighbors):
    """Return bounded KDTree indices and FP ambiguity flags for the caller policy.

    eps=0 is exact Euclidean search.  Near ties and radius boundaries are
    flagged for the caller; only small-cloud callers use a dense Torch reference.
    """
    if points.ndim != 2 or queries.ndim != 2 or points.shape[1] != 3 or queries.shape[1] != 3:
        raise ValueError('Expected [N,3] and [M,3] coordinate arrays')
    if not np.isfinite(points).all() or not np.isfinite(queries).all():
        raise ValueError('Non-finite geometry coordinates')
    k = min(neighbors, len(points))
    search_k = min(k + 1, len(points))
    tree = cKDTree(points, balanced_tree=True, compact_nodes=True)
    # Only the final int32 neighbor table scales with the entire query count.
    # Bound candidate/gather/sort temporaries independently of full-domain N.
    chunk_size=int(os.environ.get('TURBINE_GEO_TREE_QUERY_CHUNK','32768'))
    if chunk_size<=0:raise ValueError('TURBINE_GEO_TREE_QUERY_CHUNK must be positive')
    result=np.empty((len(queries),k),dtype=np.int32)
    ambiguous=np.empty(len(queries),dtype=bool)
    eps=np.finfo(points.dtype).eps;r=np.asarray(radius,dtype=points.dtype)
    for lo in range(0,len(queries),chunk_size):
        hi=min(lo+chunk_size,len(queries))
        _,ids=tree.query(queries[lo:hi],k=search_k,eps=0.,p=2.,workers=1)
        ids=np.asarray(ids).reshape(hi-lo,search_k)
        delta=points[ids]-queries[lo:hi,None,:]
        distances=np.sqrt(np.sum(delta*delta,axis=-1,dtype=points.dtype))
        order=np.argsort(distances,axis=1,kind='stable')
        distances=np.take_along_axis(distances,order,axis=1)
        ids=np.take_along_axis(ids,order,axis=1)
        tolerance=16*eps*np.maximum(distances,np.finfo(points.dtype).tiny)
        ties=np.zeros(hi-lo,dtype=bool)
        if search_k>1:
            ties=np.any(np.diff(distances,axis=1)<=np.maximum(tolerance[:,1:],tolerance[:,:-1]),axis=1)
        boundary=np.any(np.abs(distances[:,:k]-r)<=16*eps*np.maximum(np.abs(r),distances[:,:k]),axis=1)
        result[lo:hi]=np.where(distances[:,:k]<=r,ids[:,:k],-1).astype(np.int32)
        ambiguous[lo:hi]=ties|boundary
    return result,ambiguous



class NeighborIndexCache:
    def __init__(self, directory=None, max_entries=32):
        self.directory = Path(directory).expanduser().resolve() if directory else None
        self.max_entries = max(0, int(max_entries))
        self.memory = OrderedDict()
        self.memory_bytes=0
        self.max_bytes=max(0,int(os.environ.get('TURBINE_GEO_MEMORY_CACHE_MB','512')))*1024**2
        self.corruptions = 0

    def _remember(self,key,value):
        # A full-domain neighbor table can exceed 1GB; entry count alone is unsafe.
        previous=self.memory.pop(key,None)
        if previous is not None:self.memory_bytes-=previous.nbytes
        if self.max_entries and value.nbytes<=self.max_bytes:
            self.memory[key]=value;self.memory_bytes+=value.nbytes
            while len(self.memory)>self.max_entries or self.memory_bytes>self.max_bytes:
                _,old=self.memory.popitem(last=False);self.memory_bytes-=old.nbytes
        return value

    def get(self, key, shape, n_points):
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key], 'memory'
        if self.directory:
            path = self.directory / key[:2] / (key + '.npz')
            if path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as data:
                        if str(data['key']) != key or str(data['schema']) != SCHEMA:
                            raise ValueError('Cache identity mismatch')
                        indices = data['indices']
                    if indices.shape != tuple(shape) or indices.dtype != np.dtype('int32'):
                        raise ValueError('Cache shape/dtype mismatch')
                    if np.any(indices < -1) or np.any(indices >= n_points):
                        raise ValueError('Cache index out of bounds')
                    return self._remember(key, indices), 'disk'
                except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile):
                    self.corruptions += 1
        return None, None

    def put(self, key, value):
        value = np.ascontiguousarray(value, dtype=np.int32)
        if self.directory:
            directory = self.directory / key[:2]
            directory.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=directory, prefix=key + '.', suffix='.tmp', delete=False) as f:
                    temporary = Path(f.name)
                    # ZIP_STORED: avoid compression CPU overhead. ZIP CRC validates reads.
                    np.savez(f, indices=value, key=np.asarray(key), schema=np.asarray(SCHEMA))
                    f.flush()
                os.replace(temporary, directory / (key + '.npz'))
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
        self._remember(key, value)
