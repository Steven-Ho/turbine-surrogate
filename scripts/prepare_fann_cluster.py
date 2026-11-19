"""Prepare faNN subdomains and disjoint geometry splits from raw HF Parquet files."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,default=Path.home()/'Dataset/faNN')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cells',type=int,default=512)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shards=sorted((a.source/'data').glob('hf_train-*.parquet'))
 if not shards:raise RuntimeError('No raw hf_train Parquet shards under source/data')
 rows=[]
 for shard in shards:
  for i in range(pq.ParquetFile(shard).metadata.num_rows):
   dest=a.output/f'{shard.stem}_{i}.npz'
   subprocess.run([sys.executable,'scripts/prepare_fann.py','--source',str(a.source),'--parquet',str(shard.relative_to(a.source)),
       '--row',str(i),'--cells',str(a.cells),'--output',str(dest)],check=True)
   with np.load(dest) as data:geometry=int(data['geometry_id'].reshape(-1)[0])
   rows.append({'path':str(dest.resolve()),'geometry':geometry})
 geometries=sorted({r['geometry'] for r in rows})
 if len(geometries)<3:raise RuntimeError('Need at least three geometries for disjoint splits')
 rng=np.random.default_rng(42);rng.shuffle(geometries)
 n=max(1,int(len(geometries)*.1));groups={'test':set(geometries[:n]),'valid':set(geometries[n:2*n]),'train':set(geometries[2*n:])}
 result={k:[r['path'] for r in rows if r['geometry'] in g] for k,g in groups.items()}
 result.update(geometry_split={k:sorted(g) for k,g in groups.items()},cells=a.cells,protocol='first_cells_subdomain',source=str(a.source.resolve()))
 (a.output/'splits.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Prepared:',a.output/'splits.json')
if __name__=='__main__':main()
