"""Opt-in communication logging and per-rank resource snapshots."""
import json
import os
from pathlib import Path
import re
import socket
import time

ENV_NAMES=('NCCL_SOCKET_IFNAME','MCCL_SOCKET_IFNAME','NCCL_IB_HCA','MCCL_IB_HCA',
 'NCCL_IB_DISABLE','MCCL_IB_DISABLE','NCCL_NET','MCCL_NET','NCCL_NET_GDR_LEVEL','MCCL_NET_GDR_LEVEL',
 'NCCL_IB_GID_INDEX','MCCL_IB_GID_INDEX','NCCL_P2P_DISABLE','MCCL_P2P_DISABLE',
 'NCCL_SHM_DISABLE','MCCL_SHM_DISABLE','NCCL_ALGO','MCCL_ALGO',
 'OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES','MACA_VISIBLE_DEVICES',
 'SLURM_CPUS_PER_TASK','SLURM_CPU_BIND','SLURM_MEM_BIND','LOCAL_WORLD_SIZE',
 'NCCL_NET_PLUGIN','MCCL_NET_PLUGIN','GLOO_SOCKET_IFNAME','TORCH_DISTRIBUTED_DEBUG',
 'TORCH_NCCL_BLOCKING_WAIT','CUDA_LAUNCH_BLOCKING','MACA_LAUNCH_BLOCKING','MACA_LAUNCH_MODE',
 'MACA_MPS_MODE','FORCE_ACTIVE_WAIT','OMP_PROC_BIND','OMP_PLACES','KMP_AFFINITY',
 'NCCL_DEBUG','MCCL_DEBUG','NCCL_DEBUG_SUBSYS','MCCL_DEBUG_SUBSYS','NCCL_DEBUG_FILE','MCCL_DEBUG_FILE')


def read_text(path):
    try:return Path(path).read_text().strip()
    except OSError:return None


def setup_debug():
    root=os.environ.get('FANN_COMM_DIAGNOSTICS_DIR')
    if not root:return None
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    tag=f"rank_{int(os.environ.get('RANK','0')):05d}_{socket.gethostname()}_{os.getpid()}"
    for prefix in ('NCCL','MCCL'):
        os.environ[prefix+'_DEBUG']='INFO'
        os.environ[prefix+'_DEBUG_SUBSYS']='INIT,NET,GRAPH,ENV'
        os.environ[prefix+'_DEBUG_FILE']=str(root/(prefix.lower()+'_'+tag+'.log'))
    return root


def snapshot(root,device,phase):
    if root is None:return
    import torch
    import torch.distributed as dist
    rank=dist.get_rank()
    data={'phase':phase,'time_unix':time.time(),'rank':rank,'local_rank':os.environ.get('LOCAL_RANK'),
          'host':socket.gethostname(),'pid':os.getpid(),'backend':dist.get_backend(),
          'device':str(device),'torch_version':torch.__version__,
          'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
          'torch_threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
          'environment':{key:os.environ[key] for key in ENV_NAMES if key in os.environ}}
    if device.type=='cuda':
        props=torch.cuda.get_device_properties(device)
        data['gpu']={'name':props.name,'total_memory':props.total_memory,
                     'pci_bus_id':getattr(props,'pci_bus_id',None)}
    data['interfaces']={path.name:{'state':read_text(path/'operstate'),'speed_Mbps':read_text(path/'speed'),
                                  'numa_node':read_text(path/'device/numa_node')}
                        for path in Path('/sys/class/net').glob('*')}
    data['infiniband']={str(path):{'state':read_text(path/'state'),'rate':read_text(path/'rate'),
                                 'physical_state':read_text(path/'phys_state')}
                       for path in Path('/sys/class/infiniband').glob('*/ports/*')}
    maps=read_text('/proc/self/maps') or ''
    data['communication_libraries']=sorted({line.split()[-1] for line in maps.splitlines()
                                          if any(key in line.lower() for key in ('libmccl','libnccl','libmaca'))})
    tasks=list(Path('/proc/self/task').glob('*'));data['thread_count']=len(tasks);data['thread_affinities']={}
    for task in tasks[:128]:
        for line in (read_text(task/'status') or '').splitlines():
            if line.startswith('Cpus_allowed_list:'):data['thread_affinities'][task.name]=line.split(':',1)[1].strip()
    (root/f'runtime_rank_{rank:05d}_{phase}.json').write_text(json.dumps(data,indent=2)+'\n')


def summarize_communication(folder):
    folder=Path(folder);root=folder/'communication'
    files=sorted(root.glob('*.log'));evidence=[];bootstrap=[]
    pattern=re.compile(r'NET/|Using network|HCA|GDR|SHM|P2P|WARN|network plugin',re.I)
    for path in files+[folder/'console.log']:
        if not path.exists():continue
        with path.open(errors='replace') as stream:
            for number,line in enumerate(stream,1):
                if not pattern.search(line) and 'Bootstrap' not in line:continue
                target=bootstrap if 'Bootstrap' in line else evidence
                if len(target)<300:target.append({'file':str(path),'line':number,'text':line.rstrip()})
    result={'log_files':[str(path) for path in files],'transport_evidence':evidence,'bootstrap_evidence':bootstrap,
            'runtime_snapshots':[str(path) for path in sorted(root.glob('runtime_*.json'))],
            'note':'Bootstrap socket lines do not prove data transport uses sockets. No RDMA/TCP classification inferred automatically.'}
    path=folder/'communication_summary.json';path.write_text(json.dumps(result,indent=2)+'\n')
    return str(path)
