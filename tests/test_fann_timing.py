import json
import unittest
from tests.test_fann_ddp import DDPTests


class TimingTests(unittest.TestCase):
    def test_geo_bf16_rank_phases_tail_checkpoint_and_validation(self):
        case=DDPTests();case.setUp()
        try:
            case.env.update(CUDA_VISIBLE_DEVICES='',FANN_PROFILE_STEPS='2',TURBINE_GEO_NODE_CHUNK='7',
                TURBINE_GEO_ACTIVATION_OFFLOAD='0',TURBINE_GEO_CACHE_DIR=str(case.root/'neighbors'))
            result=case.run_job('GeoTransolver',case.root/'run',['--amp','bf16','--max-updates','3'])
            rows=[json.loads(line) for line in (case.root/'run/GeoTransolver/timings.jsonl').read_text().splitlines()]
            steps=[row for row in rows if row['event']=='train_step']
            self.assertEqual(len(steps),2)
            self.assertTrue(any(row['event']=='evaluation' for row in rows))
            self.assertTrue(any(row['event']=='checkpoint' for row in rows))
            self.assertIn('DDP timing',result.stdout)
            for row in rows:
                self.assertEqual(len(row['ranks']),2)
                for stats in row['summary'].values():
                    self.assertGreaterEqual(stats['min'],0)
                    self.assertLessEqual(stats['min'],stats['mean'])
                    self.assertLessEqual(stats['mean'],stats['max'])
            self.assertIn('source',steps[0]['ranks'][0])
            self.assertTrue(steps[0]['ranks'][0]['geo']['queries'])
            active=steps[0]['ranks'][0]['seconds']
            for key in ['load_sample','prepare_input','forward','target_loss_or_idle','loss_collectives','backward_with_ddp_sync','clip_optimizer','step_total']:
                self.assertIn(key,active)
            self.assertFalse(steps[1]['ranks'][1]['active'])
        finally:case.doCleanups()
