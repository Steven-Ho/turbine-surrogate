"""Tiny GPU/CPU correctness gate for portable Geo node-domain collectives.

Launch with torchrun. All physical ranks cooperate on one deterministic case.
This checks capability/numerics, not real-faNN throughput.
"""
import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch_geometric.data import Data
from tfai.models.transolver_adapters import GeoTransolver
from tfai.models._vendor.geotransolver2.domain_parallel import Domain
from tfai.benchmarks.fann_training import atomic_json
from tfai.benchmarks.fann_logging import print
from tfai.benchmarks.fann_comm_diagnostics import snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--backend',default='nccl')
    parser.add_argument('--scale',choices=['small','base','large'],default='small')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda':
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; no CPU fallback')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    else:device=torch.device('cpu')
    dist.init_process_group(args.backend)
    try:
        domain=Domain(dist.get_world_size())
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/'runtime').mkdir(exist_ok=True)
        snapshot(args.output/'runtime',device,'domain_probe')
        results=[]
        for amp in ['off','bf16']:
            torch.manual_seed(42)
            config={'model':{'scale':args.scale,'geotransolver_backend':'portable_torch'}}
            reference=GeoTransolver(config,6,6).to(device)
            sharded=GeoTransolver(config,6,6).to(device)
            sharded.load_state_dict(reference.state_dict())
            for model in [reference,sharded]:
                for module in model.modules():module.node_chunk_size=16
            domain.attach(sharded)
            x=torch.rand(67,6,device=device);x[:,3]=.1;x[:,-2:]=.2
            target=torch.randn(67,6,device=device)
            reference.train();sharded.train()
            opt1=torch.optim.Adam(reference.parameters(),lr=1e-4)
            opt2=torch.optim.Adam(sharded.parameters(),lr=1e-4)
            def context():return torch.autocast(device_type=device.type,dtype=torch.bfloat16) if amp=='bf16' else nullcontext()
            with context():full=reference(Data(x=x))
            (full.float()-target).square().mean().backward()
            with context():part=sharded(Data(x=x))
            loss=(part.float()-domain.select(target)).square().sum()/target.numel()
            loss.backward()
            # Domain parameter replicas SUM contributions, without dividing by
            # domain size. No data replicas in this standalone capability gate.
            for p in sharded.parameters():
                if p.grad is not None:dist.all_reduce(p.grad,group=domain.group)
            grad_error=grad_norm=0.
            for a,b in zip(reference.parameters(),sharded.parameters()):
                if a.grad is not None:
                    if b.grad is None:raise RuntimeError('Missing distributed gradient')
                    grad_error+=float((a.grad-b.grad).double().square().sum())
                    grad_norm+=float(a.grad.double().square().sum())
            out_error=float((part.float()-domain.select(full.detach().float())).abs().max())
            grad_relative=(grad_error/max(grad_norm,1e-30))**.5
            for model in [reference,sharded]:torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt1.step();opt2.step()
            update_error=max(float((a-b).detach().abs().max()) for a,b in zip(reference.parameters(),sharded.parameters()))
            sharded.eval()
            with torch.inference_mode(),context():
                gathered=sharded(Data(x=x))
            if gathered.shape!=(67,6) or not torch.isfinite(gathered).all():
                raise RuntimeError('Domain evaluation all_gather failed')
            values=torch.tensor([out_error,grad_relative,update_error],device=device)
            dist.all_reduce(values,op=dist.ReduceOp.MAX)
            out_error,grad_relative,update_error=values.tolist()
            passed=(out_error < (0.05 if amp=='bf16' else 2e-5)
                    and grad_relative < (0.05 if amp=='bf16' else 2e-4)
                    and update_error < (3e-4 if amp=='bf16' else 1e-5))
            result=dict(amp=amp,passed=passed,max_output_abs_error=out_error,
                        gradient_relative_l2_error=grad_relative,max_parameter_update_abs_error=update_error)
            results.append(result)
            if dist.get_rank()==0:
                print(f'Geo domain capability: {result}',flush=True)
                atomic_json(dict(scale=args.scale,world_size=dist.get_world_size(),torch=torch.__version__,backend=dist.get_backend(),
                                 device=args.device,results=results),args.output/'capability.json')
            if not passed:raise RuntimeError(f'Domain capability/numerical check failed: {result}')
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
