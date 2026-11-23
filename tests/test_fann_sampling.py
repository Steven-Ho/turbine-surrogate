import json
import math
import os
import numpy as np
import unittest
from pathlib import Path
import torch
from tfai.benchmarks.fann_sampling import geometry_batches,geometry_owners
from tests.test_fann_ddp import DDPTests


class SamplingTests(unittest.TestCase):
    def test_sampler_resume_identity_and_graph_schema_compatibility(self):
        from tfai.benchmarks.fann_training import assert_resume_config,execution_config
        self.assertEqual(execution_config('GeoTransolver',{}),execution_config('GeoTransolver',{'sampler':'random'}))
        with self.assertRaisesRegex(ValueError,'sampler'):
            assert_resume_config({}, {'sampler':'geometry_affinity'})
        graph=execution_config('GCN',{'graph_execution':'bounded'})
        self.assertTrue(all(isinstance(key,str) for key in graph))
        assert_resume_config({('graph_execution_schema','amp'):graph['graph_execution_schema']},
                             {'graph_execution_schema':graph['graph_execution_schema']})

    def test_coverage_balance_determinism_and_affinity(self):
        keys=[str(i//4) for i in range(250)]
        for world in [1,2,3,16,300]:
            batches=geometry_batches(keys,world,42,0)
            self.assertEqual(len(batches),math.ceil(len(keys)/world))
            self.assertEqual(sorted(i for batch in batches for i in batch if i is not None),list(range(len(keys))))
            self.assertTrue(all(all(i is not None for i in batch) for batch in batches[:-1]))
            counts=[sum(batch[rank] is not None for batch in batches) for rank in range(world)]
            self.assertLessEqual(max(counts)-min(counts),1)
            self.assertEqual(batches,geometry_batches(keys,world,42,0))
        owners=geometry_owners(keys,16,42);batches=geometry_batches(keys,16,42,0)
        preferred=sum(owners[keys[i]]==rank for batch in batches for rank,i in enumerate(batch) if i is not None)
        self.assertGreater(preferred/len(keys),.85)
        self.assertNotEqual(batches,geometry_batches(keys,16,42,1))
        # No huge group is allowed to leave other ranks idle.
        skew=['A']*97+['B']*3
        rows=geometry_batches(skew,16,42,0)
        self.assertEqual(len(rows),7)
        self.assertEqual(sum(i is not None for i in rows[-1]),4)

    def test_geo_affinity_exact_resume_and_recorded_coverage(self):
        case=DDPTests();case.setUp()
        try:
            # Two distinct operating conditions share an exact coordinate geometry.
            with np.load(case.root/'case_0.npz') as data:xyz=data['coordinates']
            with np.load(case.root/'case_1.npz') as data:sample={key:data[key] for key in data.files}
            sample['coordinates']=xyz
            np.savez_compressed(case.root/'case_1.npz',**sample)
            case.env.update(CUDA_VISIBLE_DEVICES='',TURBINE_FANN_CACHE_WORK_ROOT=str(case.root/'work'),
                TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'),TURBINE_GEO_NODE_CHUNK='7',TURBINE_GEO_ACTIVATION_OFFLOAD='0')
            extra=['--sampler','geometry_affinity','--amp','bf16']
            full=case.root/'full';resume=case.root/'resume'
            case.run_job('GeoTransolver',full,extra)
            case.run_job('GeoTransolver',resume,extra+['--max-updates','1'])
            case.run_job('GeoTransolver',resume,extra+['--resume'])
            a=torch.load(full/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            b=torch.load(resume/'GeoTransolver/last.pt',map_location='cpu',weights_only=False)
            self.assertEqual(a['geometry_keys'][0],a['geometry_keys'][1])
            self.assertEqual(a['history'],b['history'])
            self.assertEqual(a['scheduler_state_dict'],b['scheduler_state_dict'])
            for key,state in a['optimizer_state_dict']['state'].items():
                for field,value in state.items():torch.testing.assert_close(value,b['optimizer_state_dict']['state'][key][field],rtol=0,atol=0)
            for key,value in a['state_dict'].items():torch.testing.assert_close(value,b['state_dict'][key],rtol=0,atol=0)
            records=[]
            for path in (resume/'GeoTransolver/sampling').glob('*.jsonl'):
                records.extend(json.loads(line) for line in path.read_text().splitlines())
            for epoch in (1,2):
                self.assertEqual(sorted(r['sample_index'] for r in records if r['epoch']==epoch and r['active']),[0,1,2])
            self.assertTrue(all('cache_delta' in row for row in records))
            self.assertEqual(a['config']['sampler'],'geometry_affinity')
        finally:case.doCleanups()
