"""Five-field CR15 cache adapter. No use of solved diagnostics as features."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data
from tfai.models.transolver_adapters import GeoTransolver

FIELDS=['VelocityX','VelocityY','VelocityZ','Pressure','Temperature']
UNITS=['m/s','m/s','m/s','Pa','K']


def load_prepared(folder):
    folder=Path(folder).expanduser().resolve()
    summary=json.loads((folder/'summary.json').read_text())
    if summary.get('state')!='complete' or not summary.get('cache_ready'):
        raise ValueError('CR15 preparation must be complete with cache_ready=true')
    if summary.get('schema')!='cr15-map-five-fields-v1':raise ValueError('Unsupported CR15 cache schema')
    contract=json.loads((folder/'contract.json').read_text());splits=json.loads((folder/'splits.json').read_text())
    if contract['target_names']!=FIELDS:raise ValueError('Unexpected CR15 targets')
    if len(contract['boundary_names'])!=6:raise ValueError('Expected six declared boundary inputs')
    signature=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if signature!=summary['contract_signature']:raise ValueError('Prepared contract identity differs')
    statistics=json.loads((folder/'train_statistics.json').read_text())
    if statistics['source_split']!='train':raise ValueError('Normalization must use train only')
    paths={};nodes={};seen=set()
    for split in ['train','valid','test','baseline']:
        if split!='baseline' and not splits.get(split):raise ValueError('Empty '+split+' split')
        paths[split]=[];nodes[split]=[]
        if len(splits.get(split,[]))!=summary['splits'][split]:raise ValueError('Split membership count differs')
        for relative in splits.get(split,[]):
            path=(folder/relative).resolve()
            if path in seen:raise ValueError('Duplicate case across splits')
            seen.add(path)
            # Small prepared metadata only: no raw-data scan, array hash or stats pass.
            metadata=json.loads((folder/'metadata'/f'{path.stem}.json').read_text())
            if metadata['split']!=split or metadata['grade']!='A':raise ValueError('Wrong grade/split in cache')
            paths[split].append(str(path));nodes[split].append(int(metadata['nodes']))
    if sum(nodes['train'])!=statistics['nodes']:raise ValueError('Training statistics count differs')
    identity=hashlib.sha256(json.dumps(dict(contract=signature,splits=splits,statistics=statistics),sort_keys=True).encode()).hexdigest()
    return dict(root=str(folder),identity=identity,paths=paths,nodes=nodes,contract=contract,statistics=statistics)


class CR15Normalizer:
    def __init__(self, prepared):
        stats=prepared['statistics'];contract=prepared['contract']
        self.origin=np.asarray(stats['coordinate_minimum'],dtype='float32')
        self.length=max(float(np.max(np.asarray(stats['coordinate_maximum'])-self.origin)),1e-8)
        self.mean=np.asarray(stats['mean'],dtype='float32')
        self.scale=np.maximum(np.asarray(stats['std'],dtype='float32'),[1.,1.,1.,100.,1.]).astype('float32')
        self.boundaries=np.asarray(contract['boundary_values'],dtype='float32')
        self.boundary_scale=np.maximum(np.abs(self.boundaries),1.).astype('float32')
        if self.mean.shape!=(5,) or not np.isfinite(np.r_[self.origin,self.mean,self.scale,self.boundaries]).all():
            raise ValueError('Invalid train-only normalization')

    def input(self,sample,device):
        conditions=np.asarray(sample['global_conditions'],dtype='float32')
        if conditions.shape!=(6,) or not np.array_equal(conditions,self.boundaries):
            raise ValueError('Cached conditions differ from the declared fixed-boundary dataset')
        xyz=(sample['coordinates']-self.origin)/self.length
        features=np.column_stack([xyz,np.broadcast_to(conditions/self.boundary_scale,(len(xyz),6))]).astype('float32')
        return Data(x=torch.as_tensor(features,device=device))

    def target(self,sample,device):
        return torch.as_tensor((sample['targets']-self.mean)/self.scale,device=device)

    def decode(self,value):
        return value.detach().float().cpu().numpy()*self.scale+self.mean

    def state(self):
        return {key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in vars(self).items()}


class CR15Store:
    def __init__(self, paths):self.paths=paths
    def __len__(self):return len(self.paths)
    @lru_cache(maxsize=1)
    def __getitem__(self,index):
        path=Path(self.paths[index])
        with np.load(path,allow_pickle=False) as data:
            sample={key:data[key] for key in ['coordinates','targets','global_conditions','output_names','original_node_ids']}
        if sample['output_names'].tolist()!=FIELDS or sample['targets'].shape!=(len(sample['coordinates']),5):
            raise ValueError('Wrong five-field cached sample')
        sample['source']=str(path);return sample


class CR15GeoTransolver(nn.Module):
    def __init__(self,scale='base',amp='bf16',chunk=32768,offload=False):
        super().__init__()
        self.net=GeoTransolver({'model':{'scale':scale,'geotransolver_backend':'portable_torch','global_dim':6}},9,5)
        self.net.activation_offload=offload;self.amp=amp
        for module in self.net.model.modules():module.node_chunk_size=chunk
    def forward(self,inputs):
        if self.amp=='bf16':
            with torch.autocast(device_type=inputs.x.device.type,dtype=torch.bfloat16):return self.net(inputs).float()
        return self.net(inputs)
