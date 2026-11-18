"""Set the local GPU before device-mesh initialization, then run training."""
import json
import os
import socket

import torch

local_rank = int(os.environ['LOCAL_RANK'])
assert torch.cuda.is_available()
assert torch.cuda.device_count() == int(os.environ['LOCAL_WORLD_SIZE'])
torch.cuda.set_device(local_rank)
properties = torch.cuda.get_device_properties(local_rank)
print('PDE_FM_RANK_START ' + json.dumps({
    'rank': int(os.environ['RANK']),
    'world_size': int(os.environ['WORLD_SIZE']),
    'local_rank': local_rank,
    'hostname': socket.gethostname(),
    'device': properties.name,
    'uuid': str(getattr(properties, 'uuid', 'unavailable')),
    'torch': torch.__version__,
}), flush=True)

from walrus.train import main

main()
print(f'PDE_FM_RANK_TRAIN_OK rank={os.environ["RANK"]}', flush=True)
