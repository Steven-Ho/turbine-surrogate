"""MAP pairing, split preservation, no-label-input leakage, and audit reuse."""
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from scripts.prepare_cr15_map import read_map,convert

ROOT=Path(__file__).resolve().parents[1]


def record(f, data, endian):
    marker=struct.pack(endian+'i',len(data));f.write(marker);f.write(data);f.write(marker)


def write_map(path, blocks, shapes, grid, endian='<', flags=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:
        record(f,struct.pack(endian+'i',len(blocks)),endian)
        dims=[tuple(s) if grid else (*s,8) for s in shapes]
        record(f,np.asarray(dims,dtype=endian+'i4').tobytes(),endian)
        for i,values in enumerate(blocks):
            data=np.asarray(values.T,dtype=endian+'f8').tobytes()
            if flags and flags[i] is not None:data+=np.asarray(flags[i],dtype=endian+'i4').tobytes()
            record(f,data,endian)


class CR15PreparationTests(unittest.TestCase):
    def test_multiblock_conversion_audit_reuse_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);dataset=root/'dataset';dataset.mkdir();audit=root/'audit';audit.mkdir()
            shapes=[(2,2,2),(3,1,2)];rows=[];eligible=[];truth={};source_files=[]
            for i,split in enumerate(['train','validation','test','baseline','train']):
                name=f'CR15S_{i+1:04d}';level='A' if i<4 else 'B'
                case=dataset/'CR15S/3dmap'/name;grid=case/'output_grid/ofmeshplt.xyz';solution=case/'solution/ofpvmain-pt001.p3d'
                row=dict(sample=name,stream='v1',family='synthetic',split=split,level=level,
                         solution_path=str(solution.relative_to(dataset)),m_in_kg_s='9999',pr='0.01',
                         ps_out_pa='99999999',tt_out_k='99999',beta2e_deg='2',mesh_config='0.6')
                rows.append(row)
                if level=='B':continue # Excluded B files must never be accessed.
                coords=[];values=[]
                for b,shape in enumerate(shapes):
                    n=np.prod(shape);v=np.arange(n,dtype=float)
                    coords.append(np.column_stack([.4+v*.01+i,v*.001,b+v*.002]))
                    values.append(np.column_stack([1e6+v+i*10000,10+v+i*100,-20+v,.1*v,1800-v+i*50,v*0+5,v*0+.6,v*0+1.3]))
                truth[name]=np.concatenate(values)[:,[1,2,3,0,4]]
                flags=[np.array([0,1,-1,3,0,1,-1,3],dtype='int32'),None]
                endian='>' if i%2 else '<'
                write_map(grid,coords,shapes,True,endian,flags);write_map(solution,values,shapes,False,endian)
                eligible.append(dict(sample=name,split=split,grid_path=str(grid),solution_path=str(solution),
                                     grid_sha256='a'*64,solution_sha256='b'*64))
                source_files.extend([grid,solution])
            for path,data in [(dataset/'dataset_manifest.csv',rows),(audit/'eligible_A.csv',eligible)]:
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            (audit/'summary.json').write_text(json.dumps(dict(total=5,passed=5,failed=0,eligible_A=4)))
            out=root/'prepared';command=[sys.executable,'scripts/prepare_cr15_map.py','--root',str(dataset),
                '--audit',str(audit),'--output',str(out),'--workers','2']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
            result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            summary=json.loads((out/'summary.json').read_text());self.assertTrue(summary['cache_ready'])
            self.assertEqual(summary['splits'],dict(train=1,valid=1,test=1,baseline=1))
            self.assertFalse((out/'samples/CR15S_0005.npz').exists())
            splits=json.loads((out/'splits.json').read_text());self.assertEqual(splits['valid'],['samples/CR15S_0002.npz'])
            with np.load(out/'samples/CR15S_0001.npz',allow_pickle=False) as f:
                np.testing.assert_allclose(f['targets'],truth['CR15S_0001'],rtol=1e-7)
                self.assertEqual(f['targets'].shape,(14,5));self.assertEqual(f['block_offsets'].tolist(),[0,8,14])
                self.assertEqual(f['grid_integer_flags'][:8].tolist(),[0,1,-1,3,0,1,-1,3])
                self.assertEqual(int(f['grid_flags_present'].sum()),8)
                self.assertEqual(f['global_conditions'].tolist(),[2500000.,1850.,0.,np.float32(.014).item(),1314400.,0.])
                self.assertFalse(any(k in f.files for k in ['m_in_kg_s','pr','ps_out_pa','Density']))
            stats=json.loads((out/'train_statistics.json').read_text())
            np.testing.assert_allclose(stats['mean'],truth['CR15S_0001'].mean(0));self.assertEqual(stats['nodes'],14)
            for path in source_files:path.unlink()
            # Audit is trusted: reusable cases should not touch disappeared source data.
            result=subprocess.run(command+['--resume'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stderr);self.assertEqual(result.stdout.count(': reused,'),4)

    def test_input_mesh_and_truncated_record_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'invalid.xyz';path.write_bytes(b'\x04\0\0\0bad')
            with self.assertRaises(ValueError):read_map(path,True)
            root=Path(d)
            row={'sample':'CR15S_0001','solution_path':'case/solution.p3d'}
            audit={'grid_path':str(root/'inmeshcfd.xyz'),'solution_path':str(root/'case/solution.p3d'),
                   'grid_sha256':'a'*64,'solution_sha256':'b'*64}
            with self.assertRaisesRegex(ValueError,'Only audited output'):
                convert((row,audit,{},'signature',root,root,[],False))

    def test_grade_membership_and_audit_split_mismatch_rejected(self):
        from scripts.prepare_cr15_map import selection
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);audit=root/'audit';audit.mkdir()
            with (root/'dataset_manifest.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample','level','split']);w.writeheader()
                w.writerow(dict(sample='CR15S_0001',level='A',split='test'))
            (audit/'summary.json').write_text(json.dumps(dict(total=1,passed=1,failed=0)))
            with (audit/'eligible_A.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['sample','split']);w.writeheader()
                w.writerow(dict(sample='CR15S_0001',split='train'))
            with self.assertRaisesRegex(ValueError,'Audit split differs'):selection(root,audit)
