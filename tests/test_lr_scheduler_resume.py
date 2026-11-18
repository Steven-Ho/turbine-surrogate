import copy
import unittest
import torch
from tfai.training.optimization import build_lr_scheduler, restore_lr_scheduler


class SchedulerResumeTests(unittest.TestCase):
    def make(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.Adam([parameter], lr=1e-4)
        scheduler = build_lr_scheduler(optimizer, {"lr_scheduler": {
            "enabled": True, "patience": 1, "cooldown": 1, "min_lr": 1e-6}})
        return optimizer, scheduler

    def test_plateau_resume_preserves_next_lr_changes(self):
        optimizer, scheduler = self.make()
        for loss in [1., .9, .9]: scheduler.step(loss)
        state = {"optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                 "lr_scheduler_state_dict": copy.deepcopy(scheduler.state_dict())}
        other_optimizer, other_scheduler = self.make()
        other_optimizer.load_state_dict(state["optimizer_state_dict"])
        restore_lr_scheduler(other_scheduler, state)
        for loss in [.9, .9, .8, .8, .8, .8]:
            scheduler.step(loss); other_scheduler.step(loss)
            self.assertEqual(optimizer.param_groups[0]['lr'], other_optimizer.param_groups[0]['lr'])
            self.assertEqual(scheduler.state_dict(), other_scheduler.state_dict())
        self.assertLess(optimizer.param_groups[0]['lr'], 1e-4)

    def test_upgrade_initializes_best_without_historical_lr_decay(self):
        optimizer, scheduler = self.make()
        restore_lr_scheduler(scheduler, {"best_score": .001})
        self.assertEqual(scheduler.best, .001)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual(scheduler.num_bad_epochs, 0)

    def test_minimum_lr_and_disabled_resume_guard(self):
        optimizer, scheduler = self.make()
        for _ in range(100): scheduler.step(1.)
        self.assertEqual(optimizer.param_groups[0]['lr'], 1e-6)
        self.assertIsNone(build_lr_scheduler(optimizer, {}))
        with self.assertRaises(ValueError):
            restore_lr_scheduler(None, {"lr_scheduler_state_dict": scheduler.state_dict()})
