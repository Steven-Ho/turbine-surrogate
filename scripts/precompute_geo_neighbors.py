"""Read existing processed train/valid geometries; populate the persistent cache."""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',required=True)
    parser.add_argument('--record',required=True)
    args=parser.parse_args()
    import torch
    import yaml
    from tfai.dataset.full_sample_dataset import FullSampleDataset
    from tfai.models._vendor.geotransolver2.ball_query import BQWarp
    root=Path(os.environ['TFAI_PROJECT_ROOT'])
    source=Path(args.source_run)
    if not source.is_absolute():source=root/source
    model_dir=source/'GeoTransolver'
    cfg_path=model_dir/'configs/stage_20.yaml'
    if not cfg_path.exists():cfg_path=root/'configs/cluster_long/GeoTransolver.yaml'
    cfg=yaml.safe_load(cfg_path.read_text())
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert os.environ.get('TURBINE_GEO_CACHE_DIR'), 'Persistent cache directory must be set'
    layers=[BQWarp(.05,8,backend='kdtree').cuda(),BQWarp(.25,32,backend='kdtree').cuda()]
    seen=set();started=time.perf_counter()
    record=Path(args.record);record.parent.mkdir(parents=True,exist_ok=True)
    for split in ('train','valid'):
        directory=model_dir/'data/processed'
        table=directory/f'effective_{split}.csv'
        if not table.exists():table=directory/f'{split}.csv'
        dataset=FullSampleDataset.from_csv(str(table),cfg,str(directory),phase=split)
        for i in range(len(dataset)):
            path=dataset.file_paths[i]
            if path in seen:continue
            seen.add(path)
            graph,_=dataset[i]
            points=graph.x[:,:3].contiguous().cuda()
            for layer in layers:layer._indices(points,points)
            if len(seen)%25==0:
                torch.cuda.synchronize()
                print(f'Precomputed {len(seen)} samples; seconds={time.perf_counter()-started:.1f}',flush=True)
    torch.cuda.synchronize()
    value={'samples':len(seen),'seconds':time.perf_counter()-started,
           'source':str(source),'cache_directory':os.environ['TURBINE_GEO_CACHE_DIR'],
           'layers':[{'radius':m.radius,'neighbors':m.neighbors_in_radius,'tree_builds':m.tree_builds,
                      'disk_cache_hits':m.disk_cache_hits,'reference_chunks':m.reference_chunks,
                      'ambiguity_rows':m.ambiguity_rows} for m in layers]}
    record.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2),flush=True)


if __name__=='__main__':main()
