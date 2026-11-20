"""Immutable, train-only POD basis artifacts shared by new experiment runs."""
import fcntl
import hashlib
import json
import os
import shutil
from pathlib import Path
import tempfile
import torch
from .fann_startup_cache import _default
from tfai.utils.torch_utils import atomic_torch_save

SCHEMA='fann-pod-basis-v1'


def basis_key(train,norm,rank):
    if not getattr(train,'fingerprints',None):return None
    payload={'schema':SCHEMA,'rank':int(rank),'train':list(zip(train.paths,train.fingerprints)),
             'normalization':vars(norm)}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,default=_default).encode()).hexdigest()


def shared_pod_basis(train,norm,rank,cache_dir,scratch=None,checkpoint=None):
    from .fann_training import fit_streaming_pod
    key=basis_key(train,norm,rank)
    if key is None:return fit_streaming_pod(train,norm,rank,scratch)
    root=Path(cache_dir);root.mkdir(parents=True,exist_ok=True);path=root/(key+'.pt')
    with (root/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            try:
                saved=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
                basis=saved['basis'];ids=saved['node_ids']
                if saved['schema']!=SCHEMA or saved['key']!=key or basis.ndim!=2 or basis.shape[0]!=len(ids)*6 or not 0<basis.shape[1]<=rank:
                    raise ValueError('Invalid POD cache identity/shape')
                print(f'POD: shared basis cache hit {path}',flush=True)
                return {'basis':basis,'node_ids':ids,'cache_path':str(path),'cache_status':'hit'}
            except (OSError,ValueError,RuntimeError,KeyError,EOFError):pass
        for orphan in root.glob(key+'.build-*'):
            if orphan.is_dir() and not orphan.is_symlink():shutil.rmtree(orphan)
        with tempfile.TemporaryDirectory(prefix=key+'.build-',dir=root) as directory:
            if checkpoint is not None:
                saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
                if saved.get('model')!='POD':raise ValueError('Expected a formal POD checkpoint containing a basis')
                rows=saved['manifest']['splits']['train']
                if [(row['path'],row['fingerprint']) for row in rows]!=list(zip(train.paths,train.fingerprints)) or saved['config']['rank']!=rank:
                    raise ValueError('POD checkpoint training cases/rank differ')
                if json.dumps(saved['normalization'],sort_keys=True,default=_default)!=json.dumps(vars(norm),sort_keys=True,default=_default):
                    raise ValueError('POD checkpoint normalization differs')
                state={'basis':saved['state_dict']['basis'],'node_ids':saved['state_dict']['node_ids']}
            else:
                state=fit_streaming_pod(train,norm,rank,directory,progress=lambda message:print(message,flush=True))
            stage=Path(directory)/'artifact.pt'
            atomic_torch_save({'schema':SCHEMA,'key':key,**state},stage)
            os.replace(stage,path)
        saved=torch.load(path,map_location='cpu',mmap=True,weights_only=True)
        print(f'POD: shared basis cache built {path}',flush=True)
        return {'basis':saved['basis'],'node_ids':saved['node_ids'],'cache_path':str(path),'cache_status':'built'}
