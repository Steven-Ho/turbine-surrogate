import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tfai.models._vendor.geotransolver2 import GeoTransolver
from tfai.models._vendor.geotransolver2.ball_query import BQWarp

KWARGS=dict(functional_dim=3,out_dim=1,geometry_dim=3,global_dim=2,n_layers=1,n_hidden=16,
            n_head=2,slice_num=4,mlp_ratio=2,dropout=0.,use_te=False,
            include_local_features=True,radii=[.05,.25],neighbors_in_radius=[8,32],n_hidden_local=8)

class GeometryTests(unittest.TestCase):
    def test_radius_boundary_padding_cache_and_gradients(self):
        q=BQWarp(radius=.25,neighbors_in_radius=4,query_chunk_size=1)
        x=torch.tensor([[[0.,0.,0.],[.25,0.,0.],[2.,0.,0.]]],requires_grad=True)
        keys=torch.tensor([[[0.,0.,0.],[3.,0.,0.]]])
        idx,out=q(x,keys)
        self.assertEqual(out.shape,(1,2,4,3))
        torch.testing.assert_close(out[0,0,1],x[0,1])
        self.assertEqual(torch.count_nonzero(out[0,1]).item(),0)
        out.sum().backward()
        torch.testing.assert_close(x.grad[0,0],torch.ones(3))
        torch.testing.assert_close(x.grad[0,1],torch.ones(3))
        q(x.detach().clone(),keys.clone())
        self.assertEqual(q.cache_hits,1)
        changed=x.detach().clone();changed[0,0,0]=.01
        q(changed,keys)
        self.assertEqual(q.cache_misses,2)
        self.assertEqual(q.state_dict(),{})

    def test_chunked_query_matches_full_torch_reference(self):
        torch.manual_seed(17)
        x=torch.randn(1,40,3)
        y=torch.randn(1,19,3)
        q=BQWarp(radius=1.,neighbors_in_radius=8,query_chunk_size=3)
        idx,out=q(x,y)
        d=torch.cdist(x[0],y[0],compute_mode='donot_use_mm_for_euclid_dist')
        v,i=torch.topk(d,k=8,dim=0,largest=False)
        valid=v<=1.
        expected_idx=torch.where(valid,i,0).T
        expected=x[0][expected_idx]*valid.T.unsqueeze(-1)
        torch.testing.assert_close(idx[0],expected_idx)
        torch.testing.assert_close(out[0],expected)

    def test_model_forward_backward_and_checkpoint_keys(self):
        torch.manual_seed(4)
        model=GeoTransolver(**KWARGS)
        xyz=torch.randn(1,32,3,requires_grad=True)
        cond=torch.randn(1,1,2,requires_grad=True)
        out=model(xyz,local_positions=xyz,geometry=xyz,global_embedding=cond)
        self.assertEqual(out.shape,(1,32,1))
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(xyz.grad).all())
        self.assertTrue(torch.isfinite(cond.grad).all())
        clone=GeoTransolver(**KWARGS)
        clone.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(clone(xyz.detach(),local_positions=xyz.detach(),geometry=xyz.detach(),global_embedding=cond.detach()),out.detach())

    @unittest.skipUnless(importlib.util.find_spec('physicsnemo') is not None,'Official PhysicsNeMo only needed for local parity audit')
    def test_original_architecture_outputs_and_parameter_gradients(self):
        from physicsnemo.experimental.models.geotransolver import GeoTransolver as Original
        from physicsnemo.nn.functional.radius_search._torch_impl import radius_search
        torch.manual_seed(11)
        official=Original(**KWARGS)
        port=GeoTransolver(**KWARGS)
        self.assertEqual(set(official.state_dict()),set(port.state_dict()))
        port.load_state_dict(official.state_dict(),strict=True)
        xyz=torch.randn(1,32,3)*.2
        cond=torch.randn(1,1,2)
        x1=xyz.clone().requires_grad_();x2=xyz.clone().requires_grad_()
        c1=cond.clone().requires_grad_();c2=cond.clone().requires_grad_()
        with patch('physicsnemo.nn.module.ball_query.radius_search',radius_search):
            expected=official(x1,local_positions=x1,geometry=x1,global_embedding=c1)
        actual=port(x2,local_positions=x2,geometry=x2,global_embedding=c2)
        torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
        expected.square().mean().backward();actual.square().mean().backward()
        torch.testing.assert_close(x2.grad,x1.grad,atol=3e-6,rtol=3e-5)
        torch.testing.assert_close(c2.grad,c1.grad,atol=3e-6,rtol=3e-5)
        for (name,p),(other,q) in zip(official.named_parameters(),port.named_parameters()):
            self.assertEqual(name,other)
            if p.grad is not None:
                torch.testing.assert_close(p.grad,q.grad,atol=3e-6,rtol=3e-5)

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
