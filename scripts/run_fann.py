"""Train/evaluate the seven faNN submesh adapters; explicit smoke permits one case."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from tfai.benchmarks.fann import MODELS,FIELDS,load_sample,Normalizer,FannAdapter


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train',nargs='+',required=True)
    p.add_argument('--valid',nargs='+')
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--grid',type=int,default=32)
    a=p.parse_args()
    if a.steps<1: p.error('--steps must be positive')
    if not a.smoke and not a.valid: p.error('Provide held-out --valid or explicitly use --smoke')
    if a.valid and set(map(lambda x:str(Path(x).resolve()),a.train)) & set(map(lambda x:str(Path(x).resolve()),a.valid)):
        p.error('Training and validation paths overlap')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(a.threads)
    train=[load_sample(x) for x in a.train];valid=[load_sample(x) for x in (a.valid or a.train)]
    if a.valid and {s['fingerprint'] for s in train} & {s['fingerprint'] for s in valid}:
        raise ValueError('Duplicate case content across train/validation')
    norm=Normalizer(train)
    report={'mode':'single_sample_smoke' if a.smoke else 'submesh_training_validation',
            'device':a.device,'torch':torch.__version__,'output_names':FIELDS,
            'train':[s['source'] for s in train],'valid':[s['source'] for s in valid],
            'results':{},'errors':{}}
    for name in a.models:
        print('RUN',name,flush=True);torch.manual_seed(42)
        try:
            model=FannAdapter(name,train,norm,grid=a.grid).to(a.device)
            prepared=[model.prepare(norm.graph(s,a.device),s['original_node_ids']) for s in train]
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
            before=next(model.parameters()).detach().clone();losses=[];start=time.perf_counter()
            model.train()
            for step in range(a.steps):
                i=step%len(train);optimizer.zero_grad();pred=model(prepared[i]);target=norm.target(train[i],a.device)
                assert pred.shape==target.shape and torch.isfinite(pred).all()
                loss=(pred-target).square().mean();loss.backward()
                assert any(v.grad is not None for v in model.parameters())
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
                optimizer.step();losses.append(loss.item())
            assert not torch.equal(before,next(model.parameters()).detach())
            if a.device.startswith('cuda'):torch.cuda.synchronize()
            elapsed=time.perf_counter()-start
            folder=a.output/name;folder.mkdir()
            checkpoint={'state_dict':model.state_dict(),'normalization':vars(norm),
                        'model':name,'grid':a.grid,'fields':FIELDS,'train_sources':report['train'],
                        'note':'POD basis and normalization fitted on training samples only'}
            # Full checkpoints are optional in future runs; smoke verifies an
            # in-memory state_dict reload to avoid seven large disk artifacts.
            clone=FannAdapter(name,train,norm,grid=a.grid).to(a.device)
            clone.load_state_dict(copy.deepcopy(checkpoint['state_dict']),strict=True)
            model.eval();clone.eval();metrics=[]
            with torch.no_grad():
                for i,s in enumerate(valid):
                    inputs=model.prepare(norm.graph(s,a.device),s['original_node_ids'])
                    pred=model(inputs);torch.testing.assert_close(pred,clone(inputs))
                    physical=norm.decode(pred);assert np.isfinite(physical).all()
                    np.savez_compressed(folder/f'prediction_{i}.npz',original_node_ids=s['original_node_ids'],
                                        prediction=physical,output_names=np.array(FIELDS))
                    with np.load(folder/f'prediction_{i}.npz') as saved:
                        np.testing.assert_array_equal(saved['original_node_ids'],s['original_node_ids'])
                        np.testing.assert_array_equal(saved['prediction'],physical)
                    error=physical.astype(np.float64)-s['targets']
                    metrics.append({'rmse':np.sqrt((error**2).mean(0)).tolist(),
                                    'mae':np.abs(error).mean(0).tolist(),'nodes':len(physical)})
            if not a.smoke:torch.save(checkpoint,folder/'checkpoint.pt')
            report['results'][name]={'losses':losses,'training_seconds':elapsed,
                'state_dict_reload_verified':True,'metrics':metrics,
                'metrics_are_in_sample_debug_only':not bool(a.valid)}
            print('PASS',name,losses,flush=True)
            del clone,model,optimizer,prepared,checkpoint
            gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
        except Exception as exc:
            import traceback
            report['errors'][name]=traceback.format_exc();print('FAIL',name,exc,flush=True)
        (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['errors']:raise SystemExit(1)

if __name__=='__main__':main()
