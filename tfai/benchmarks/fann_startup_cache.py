"""Register unchanged prepared datasets once; reuse checks and train-only stats.

Cache hits check file identity/size/mtime/ctime, not whole-file contents. A changed
input invalidates the registration. Training reads retain before/after metadata
checks so an active run cannot silently mix a newly replaced input.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np

SCHEMA='fann-startup-v1'


def _default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    raise TypeError(type(value).__name__)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,default=_default).encode()).hexdigest()


def file_stamp(path):
    path=Path(path);s=path.stat()
    return {'resolved':str(path.resolve()),'device':s.st_dev,'inode':s.st_ino,
            'size':s.st_size,'mtime_ns':s.st_mtime_ns,'ctime_ns':s.st_ctime_ns}


def verify_files(dependencies):
    for path,expected in dependencies.items():
        try:actual=file_stamp(path)
        except OSError as error:raise ValueError('Registered faNN data disappeared: '+path) from error
        if actual!=expected:raise ValueError('Registered faNN data changed; rebuild startup registration: '+path)


def _dependencies(split_path):
    raw=json.loads(split_path.read_text());paths={str(split_path)};by_sample={}
    for split in ['train','valid','test']:
        for value in raw.get(split,[]):
            sample=Path(value)
            if not sample.is_absolute():sample=split_path.parent/sample
            sample=sample.absolute();deps=[str(sample)]
            with np.load(sample,allow_pickle=False) as data:
                if 'mesh_file' in data.files:
                    mesh=Path(str(data['mesh_file']))
                    if not mesh.is_absolute():mesh=sample.parent/mesh
                    deps.append(str(mesh.absolute()))
            paths.update(deps);by_sample[str(sample.resolve())]=deps
    return {path:file_stamp(path) for path in sorted(paths)},by_sample


def load_startup(split_path,allow_legacy=False,cache_dir=None,workers=1,force=False,progress=None,from_checkpoint=None):
    from .fann import Normalizer
    from .fann_training import inspect_splits
    split_path=Path(split_path).resolve()
    if force:from_checkpoint=None
    if workers<1:raise ValueError('Startup workers must be positive')
    directory=os.environ.get('TURBINE_FANN_STARTUP_CACHE','runs/fann_startup_cache_v1') if cache_dir is None else str(cache_dir)
    if directory in ('','0'):
        if progress:progress('faNN startup cache disabled; strict full registration')
        return inspect_splits(split_path,allow_legacy,progress,collect_normalization=True,workers=workers)
    identity={'schema':SCHEMA,'split_path':str(split_path),'allow_legacy':allow_legacy,'numpy':np.__version__}
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    key=_digest(identity);path=directory/(key+'.json')
    with (directory/(key+'.lock')).open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            if progress:progress('faNN startup: another process is registering this dataset; waiting for shared result')
            fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists() and not force:
            try:
                envelope=json.loads(path.read_text());payload=envelope['payload']
                if envelope['sha256']!=_digest(payload) or payload['identity']!=identity:raise ValueError('Cache integrity mismatch')
                verify_files(payload['dependencies'])
                manifest=payload['manifest'];Normalizer.from_state(manifest['training_normalization'])
                if progress:progress(f'faNN startup cache hit: {path}; no full scan or normalization pass')
                return manifest
            except (OSError,ValueError,KeyError,TypeError) as error:
                if progress:progress('faNN startup cache invalidated: '+str(error))
                # Changed registered data require fresh contents, never old stats.
                from_checkpoint=None
        if progress:progress(f'faNN registering dataset once: workers={workers}; cache={path}')
        before,by_sample=_dependencies(split_path);started=time.perf_counter()
        if from_checkpoint is None:
            manifest=inspect_splits(split_path,allow_legacy,progress,collect_normalization=True,workers=workers)
        else:
            # Explicit opt-in: the caller asserts the prepared data have not changed
            # since this already validated checkpoint. Do not infer a checkpoint.
            import copy
            import torch
            checkpoint=torch.load(from_checkpoint,map_location='cpu',weights_only=False)
            manifest=copy.deepcopy(checkpoint['manifest'])
            signed={key:manifest[key] for key in ('splits','domain','wall_distance','fixed_conditions')}
            if hashlib.sha256(json.dumps(signed,sort_keys=True).encode()).hexdigest()!=manifest['signature']:
                raise ValueError('Invalid checkpoint data signature')
            raw=json.loads(split_path.read_text())
            for split in ['train','valid','test']:
                paths=[str((Path(v) if Path(v).is_absolute() else split_path.parent/v).resolve()) for v in raw[split]]
                if paths!=[row['path'] for row in manifest['splits'][split]]:
                    raise ValueError('Checkpoint split paths/order differ; use a full registration')
            if not allow_legacy and (manifest['domain'] not in ('full','structured_patches') or not manifest['wall_distance']):
                raise ValueError('Checkpoint does not satisfy formal data protocol')
            state=Normalizer.from_state(checkpoint['normalization'])
            manifest['training_normalization']=vars(state)
            if progress:progress('faNN imported prior validation and normalization from checkpoint; caller asserts data unchanged')
        verify_files(before)
        manifest['registered_files']=before;manifest['sample_dependencies']=by_sample
        payload={'identity':identity,'dependencies':before,'manifest':manifest,
                 'origin':'full_scan' if from_checkpoint is None else 'checkpoint_asserted_unchanged',
                 'checkpoint':str(Path(from_checkpoint).resolve()) if from_checkpoint is not None else None}
        envelope={'sha256':_digest(payload),'payload':payload}
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(mode='w',dir=directory,suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name)
                json.dump(envelope,stream,allow_nan=False,default=_default)
                stream.write('\n')
            os.replace(temporary,path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
        if progress:progress(f'faNN registration complete in {time.perf_counter()-started:.1f}s; later models reuse this result')
        return manifest
