import importlib.util
import json
from pathlib import Path
import unittest
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

runner = module('run_cluster_long')
compare = module('compare_cluster_long')

class ProtocolTests(unittest.TestCase):
    def test_extension_requires_budget_and_late_improvement(self):
        for steps, loss, expected in [(200, .97, True), (200, .99, False), (199, .97, False)]:
            self.assertEqual(runner.should_extend({'val_units':[160*800,steps*800], 'val_loss':[1.,loss]}),expected)

    def test_seven_frozen_configs_and_long_schedule(self):
        import yaml
        baseline=json.loads((ROOT/'configs/cluster_long/local_baseline.json').read_text())
        for model in runner.MODELS:
            original=yaml.safe_load((ROOT/f'configs/cluster_long/{model}.yaml').read_text())
            for target in [20,200,400]:
                cfg=runner.build_stage_config(original,baseline,Path('/tmp/test-run'),target)
                self.assertEqual(cfg['model']['type'],model)
                self.assertEqual(cfg['general']['seed'],42)
                self.assertEqual(cfg['training']['learning_rate'],1e-4)
                self.assertEqual(cfg['training']['batch_size'],1)
                self.assertEqual(cfg['training']['epochs'],target)
                self.assertEqual(cfg['training']['resume']['enabled'],target>20)
                if target>20:
                    self.assertEqual(cfg['training']['early_stopping']['patience'],60)
                    self.assertEqual(cfg['training']['lr_scheduler']['min_lr'],1e-6)
                    self.assertTrue(cfg['training']['require_preprocess_cache'])
                else:
                    self.assertFalse(cfg['training'].get('lr_scheduler',{}).get('enabled',False))
            self.assertEqual(original['training']['epochs'],20,'Original snapshot mutated')

    def test_equal_and_unequal_budgets_are_distinguished(self):
        baseline={'models':{'POD':{'twenty_epoch':{'training_seconds':100},'long_stages':{'200':{'training_seconds':200}},'long_completed_epochs':163}}}
        state={'state':'complete','stages':{'20':{'completed_epochs':20,'training_seconds':50},'200':{'completed_epochs':150,'training_seconds':130}}}
        rows=compare.stage_rows('POD',state,baseline)
        self.assertEqual(rows[0]['local_over_cluster_throughput_ratio'],2)
        self.assertTrue(rows[0]['identical_epoch_budget'])
        self.assertFalse(rows[1]['identical_epoch_budget'])

    def test_missing_local_long_timing_not_fabricated(self):
        baseline={'models':{'GCN':{'twenty_epoch':{},'long_stages':{}}}}
        rows=compare.stage_rows('GCN',{'state':'complete','stages':{'200':{'completed_epochs':200,'training_seconds':10}}},baseline)
        self.assertIsNone(rows[0]['local_seconds_per_additional_epoch'])
        self.assertIsNone(rows[0]['local_over_cluster_throughput_ratio'])

    def test_submission_queues_summary_after_array_even_if_tasks_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory)
            log=d/'calls'
            fake=d/'sbatch'
            fake.write_text('#!/bin/bash\necho "$* ROOT=$TURBINE_LONG_RUN_ROOT" >> "$TEST_CALLS"\necho 12345\n')
            fake.chmod(0o755)
            env=dict(os.environ, PATH=str(d)+os.pathsep+os.environ['PATH'], TEST_CALLS=str(log), TURBINE_LONG_RUN_ROOT='runs/test_batch')
            subprocess.run(['bash',str(ROOT/'scripts/submit_cluster_long.sh')],cwd=ROOT,env=env,check=True,capture_output=True,text=True)
            calls=log.read_text().splitlines()
            self.assertEqual(len(calls),2)
            self.assertIn('--dependency=afterany:12345',calls[1])
            self.assertTrue(all('ROOT=runs/test_batch' in call for call in calls))

    def test_duration_parses_hours(self):
        self.assertEqual(runner.duration('模型训练耗时: 1 小时 2 分 3.5 秒'),3723.5)
        self.assertIsNone(runner.duration('training failed'))

if __name__=='__main__': unittest.main()
