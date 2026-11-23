"""Combine independently scheduled scaling tests; standard library only."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    root=Path(root);rows=[];parts=[];missing=[]
    for count in (1,2,4,8):
        path=root/'parts'/f'{count}gpu'/'report.json'
        if not path.exists():missing.append(count);continue
        part=json.loads(path.read_text())
        if part.get('state')!='complete':missing.append(count);continue
        if len(part.get('results',[]))!=1 or part['results'][0]['gpus']!=count:
            raise ValueError('Unexpected process-count result: '+str(path))
        parts.append(part)
        rows.append({**part['results'][0],'host':part.get('host'),'gpu_name':part.get('gpu_name'),
                     'job_id':part.get('job_id'),'part_report':str(path)})
    if parts:
        for field in ('dataset_signature','warmup_cases','measure_cases','torch_version','device','gpu_name'):
            if len({json.dumps(part.get(field),sort_keys=True) for part in parts})!=1:
                raise ValueError('Inconsistent test setting: '+field)
        for row in rows:
            if row['measured_cases']!=parts[0]['measure_cases']:raise ValueError('Measured case count mismatch')
    baseline=next((row['cases_per_second'] for row in rows if row['gpus']==1),None)
    for row in rows:
        if baseline:
            row['speedup']=row['cases_per_second']/baseline
            row['efficiency']=row['speedup']/row['gpus']
    return {'state':'incomplete' if missing else 'complete','missing_counts':missing,'results':rows,
            'note':'Independent node allocations; node variation and concurrent shared-storage reads may affect results.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
    report=summarize(args.root)
    (args.root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    columns=['gpus','host','gpu_name','job_id','measured_cases','wall_seconds','cases_per_second','speedup','efficiency','mean_rank0_seconds']
    with (args.root/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for row in report['results']:writer.writerow({key:row.get(key,'') for key in columns})
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
