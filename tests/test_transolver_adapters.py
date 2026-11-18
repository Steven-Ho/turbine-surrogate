import hashlib
import json
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Data
from tfai.models.transolver_adapters import Transolver3, GeoTransolver

ROOT=Path(__file__).resolve().parents[1]

def fixture(n=96):
    device="cuda" if torch.cuda.is_available() else "cpu"
    xyz=torch.rand(n,3,device=device)*2-1
    condition=torch.tensor([.1,.8],device=device).expand(n,2)
    return Data(x=torch.cat([xyz,condition],dim=1))

class TransolverAdapterTests(unittest.TestCase):
    def test_vendored_transolver3_is_unmodified_official_source(self):
        directory=ROOT/"tfai/models/_vendor/transolver3"
        metadata=json.loads((directory/"UPSTREAM.json").read_text())
        source=directory/Path(metadata["file"]).name
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),metadata["sha256"])
        self.assertEqual(metadata["modifications"],[])

    def test_chunked_transolver_matches_official_full_pass(self):
        graph=fixture()
        model=Transolver3({"model":{"scale":"small","transolver_chunk_size":24}},5,1).to(graph.x.device).eval()
        with torch.no_grad():
            chunked=model(graph)
            full=model.model(graph.x[None],use_checkpoint=False,input_list=False).squeeze(0)
        torch.testing.assert_close(chunked,full,atol=2e-5,rtol=2e-4)

    def test_checkpointed_training_and_target_leakage_guard(self):
        graph=fixture()
        model=Transolver3({"model":{"scale":"small","transolver_chunk_size":24}},5,1).to(graph.x.device)
        first=model(graph)
        graph.y=torch.full((graph.x.shape[0],1),1e9,device=graph.x.device)
        second=model(graph)
        torch.testing.assert_close(first,second)
        second.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    @unittest.skipUnless(torch.cuda.is_available(),"Official multiscale GeoTransolver GPU check")
    def test_official_geometry_global_multiscale_paths_and_gradients(self):
        graph=fixture()
        model=GeoTransolver({"model":{"scale":"small"}},5,1).cuda()
        self.assertTrue(model.model.include_local_features)
        first=model(graph)
        other=Data(x=graph.x.clone());other.x[:,3]=.9
        second=model(other)
        self.assertEqual(tuple(first.shape),(graph.x.shape[0],1))
        self.assertGreater(float((first-second).abs().max()),1e-8)
        second.square().mean().backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
