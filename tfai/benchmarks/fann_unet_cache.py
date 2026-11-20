"""Cache target-independent nearest-node voxel maps for repeated geometries."""
import hashlib
import os
from pathlib import Path
import tempfile
import numpy as np
from scipy.spatial import cKDTree


def mapping_path(xyz,positions,cache_dir):
    xyz=np.ascontiguousarray(xyz);positions=np.ascontiguousarray(positions)
    digest=hashlib.sha256(b'fann-unet-nearest-v1')
    for array in [xyz,positions]:
        digest.update(str((array.shape,array.dtype.str)).encode());digest.update(memoryview(array).cast('B'))
    return Path(cache_dir)/(digest.hexdigest()+'.npz')


def nearest_mapping(xyz,positions,cache_dir):
    xyz=np.ascontiguousarray(xyz);positions=np.ascontiguousarray(positions)
    directory=Path(cache_dir);directory.mkdir(parents=True,exist_ok=True)
    path=mapping_path(xyz,positions,directory)
    if path.exists():
        with np.load(path,allow_pickle=False) as data:
            ids=data['ids'];distance=data['distance']
        if ids.shape!=(len(positions),) or distance.shape!=ids.shape or ids.dtype.kind not in 'iu' or (ids<0).any() or (ids>=len(xyz)).any() or not np.isfinite(distance).all():
            raise ValueError('Invalid cached UNet mapping: '+str(path))
        return distance,ids
    workers=int(os.environ.get('TURBINE_FANN_TREE_WORKERS','1'))
    if workers<1:raise ValueError('TURBINE_FANN_TREE_WORKERS must be positive')
    distance,ids=cKDTree(xyz).query(positions,workers=workers)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=directory,suffix='.npz',delete=False) as f:
            temporary=Path(f.name);np.savez(f,distance=distance,ids=ids)
        os.replace(temporary,path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)
    return distance,ids
