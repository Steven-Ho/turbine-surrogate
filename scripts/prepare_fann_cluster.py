"""Prepare faNN subdomains and disjoint geometry splits from raw HF Parquet files."""
import argparse,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--source',type=Path,default=Path.home()/'Dataset/faNN')
 p.add_argument('--output',type=Path,required=True)
 p.add_argument('--cells',type=int,default=10000)
 p.add_argument('--selection',choices=['first','structured_patches','full'],default='structured_patches')
 p.add_argument('--seed',type=int,default=42)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 shards=sorted((a.source/'data').glob('hf_train-*.parquet'))
 if not shards:raise RuntimeError('No raw hf_train Parquet shards under source/data')
 rows=[]
 for shard in shards:
  for i in range(pq.ParquetFile(shard).metadata.num_rows):
   dest=a.output/f'{shard.stem}_{i}.npz'
   subprocess.run([sys.executable,'scripts/prepare_fann.py','--source',str(a.source),'--parquet',str(shard.relative_to(a.source)),
       '--row',str(i),'--cells',str(a.cells),'--selection',a.selection,'--seed',str(a.seed),
       '--selection-file',str(a.output/'selected_cell_ids.npy'),'--shared-mesh',str(a.output/'shared_mesh.npz'),'--output',str(dest)],check=True)
   with np.load(dest) as data:geometry=int(data['geometry_id'].reshape(-1)[0])
   rows.append({'path':str(dest.resolve()),'geometry':geometry})
 geometries=sorted({r['geometry'] for r in rows})
 if len(geometries)<3:raise RuntimeError('Need at least three geometries for disjoint splits')
 rng=np.random.default_rng(a.seed);rng.shuffle(geometries)
 n=max(1,int(len(geometries)*.1));groups={'test':set(geometries[:n]),'valid':set(geometries[n:2*n]),'train':set(geometries[2*n:])}
 result={k:[r['path'] for r in rows if r['geometry'] in g] for k,g in groups.items()}
 actual_cells=a.cells
 if a.selection=='full':
  with np.load(a.output/'shared_mesh.npz') as mesh:actual_cells=len(mesh['hexahedra'])
 result.update(geometry_split={k:sorted(g) for k,g in groups.items()},cells=actual_cells,full_domain=a.selection=='full',protocol=a.selection,seed=a.seed,format_version=2,source=str(a.source.resolve()))
 (a.output/'splits.json').write_text(json.dumps(result,indent=2)+'\n')
 print('Prepared:',a.output/'splits.json')
if __name__=='__main__':main()
