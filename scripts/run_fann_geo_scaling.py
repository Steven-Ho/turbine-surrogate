"""Sequential 1/2/4/8-process training throughput in a single node allocation."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import torch
from tfai.benchmarks.fann_startup_cache import load_startup
from tfai.benchmarks.fann_training import atomic_json

STEP=re.compile(r'GeoTransolver DDP .*? step=(\d+) epoch_cases=\d+/\d+ batch_cases=(\d+) .*?rank0_seconds=([\d.]+)')


def write_report(output,report):
    atomic_json(report,output/'report.json')
    columns=['gpus','measured_cases','wall_seconds','cases_per_second','speedup','efficiency',
             'mean_rank0_seconds','median_rank0_seconds']
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in report['results']:writer.writerow({key:row.get(key,'') for key in columns})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--splits',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-counts',nargs='+',type=int,default=[1,2,4,8])
    parser.add_argument('--warmup-cases',type=int,default=16)
    parser.add_argument('--measure-cases',type=int,default=64)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--backend',default='nccl')
    args=parser.parse_args()
    if not args.gpu_counts or len(set(args.gpu_counts))!=len(args.gpu_counts) or any(p<1 for p in args.gpu_counts):parser.error('Counts must be unique and positive')
    if min(args.warmup_cases,args.measure_cases,args.threads)<1:parser.error('Budgets and threads must be positive')
    if any(args.warmup_cases%p or args.measure_cases%p for p in args.gpu_counts):parser.error('Both case budgets must be divisible by every process count')
    if args.device=='cuda' and (not torch.cuda.is_available() or max(args.gpu_counts)>torch.cuda.device_count()):raise RuntimeError('Insufficient visible GPUs; no CPU fallback')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'state':'running','device':args.device,'gpu_counts':args.gpu_counts,'warmup_cases':args.warmup_cases,
            'measure_cases':args.measure_cases,'results':[],'torch_version':torch.__version__,
            'note':'Same shuffled case prefix, seed42 fresh runs; global batch and optimizer trajectory differ. Later runs may benefit from OS file cache.'}
    write_report(args.output,report)
    try:
        manifest=load_startup(args.splits,progress=lambda message:print(message,flush=True))
        total_cases=args.warmup_cases+args.measure_cases
        if total_cases>len(manifest['splits']['train']):raise ValueError('Case budget must fit within the first epoch to preserve identical sample order')
        report['dataset_signature']=manifest['signature'];report['train_cases']=len(manifest['splits']['train'])
        for count in args.gpu_counts:
            folder=args.output/f'gpus_{count}';folder.mkdir()
            updates=total_cases//count;warm_steps=args.warmup_cases//count
            env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(args.threads),
                MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
                FANN_DDP_SAMPLER='random',FANN_PROFILE_STEPS='0',TURBINE_FANN_PROFILE_STEPS='0',FANN_LOG_EVERY='1',
                TURBINE_FANN_PIPELINE_GPUS='1',TURBINE_GEO_ACTIVATION_OFFLOAD='0',TURBINE_GEO_NODE_CHUNK='32768',
                TURBINE_GEO_MEMORY_CACHE_MB='512',TURBINE_GEO_MEMORY_CACHE='32',
                TURBINE_GEO_QUERY_BACKEND='kdtree',TURBINE_GEO_REFERENCE_MAX_POINTS='0')
            command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={count}',
                'scripts/train_fann_ddp.py','--models','GeoTransolver','--splits',str(args.splits),
                '--output',str(folder),'--device',args.device,'--backend',args.backend,'--amp','bf16',
                '--sampler','random','--scale','small','--seed','42','--threads',str(args.threads),'--epochs','400',
                '--checkpoint-every',str(updates+1),'--max-updates',str(updates)]
            print(f'SCALING start processes={count} warmup_cases={args.warmup_cases} measure_cases={args.measure_cases} updates={updates}',flush=True)
            records=[];measure_start=measure_end=None
            with (folder/'console.log').open('w') as log:
                process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                try:
                    for line in process.stdout:
                        log.write(line);log.flush();print(line,end='',flush=True)
                        match=STEP.search(line)
                        if match:
                            step,cases,seconds=int(match[1]),int(match[2]),float(match[3]);now=time.perf_counter()
                            records.append({'step':step,'cases':cases,'rank0_seconds':seconds,'measured':step>warm_steps})
                            if step==warm_steps:measure_start=now
                            if step==updates:measure_end=now
                    result=process.wait()
                finally:
                    if process.poll() is None:process.terminate();process.wait()
            atomic_json(records,folder/'steps.json')
            if result:raise RuntimeError(f'{count}-process test failed with exit {result}; see {folder}/console.log')
            measured=[row for row in records if row['measured']]
            if len(records)!=updates or sum(row['cases'] for row in measured)!=args.measure_cases or measure_start is None or measure_end is None:
                raise RuntimeError('Incomplete or inconsistent training-step records')
            wall=measure_end-measure_start
            report['results'].append({'gpus':count,'measured_cases':args.measure_cases,'wall_seconds':wall,
                'cases_per_second':args.measure_cases/wall,
                'mean_rank0_seconds':statistics.mean(row['rank0_seconds'] for row in measured),
                'median_rank0_seconds':statistics.median(row['rank0_seconds'] for row in measured)})
            baseline=next((row['cases_per_second'] for row in report['results'] if row['gpus']==1),None)
            if baseline:
                for row in report['results']:
                    row['speedup']=row['cases_per_second']/baseline;row['efficiency']=row['speedup']/row['gpus']
            write_report(args.output,report)
            print('SCALING result '+json.dumps(report['results'][-1]),flush=True)
        report['state']='complete';write_report(args.output,report)
    except BaseException as error:
        report.update(state='failed',error=str(error));write_report(args.output,report);raise


if __name__=='__main__':main()
