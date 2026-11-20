import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
import torch
from tfai.benchmarks.fann import FannAdapter,Normalizer
from tfai.benchmarks.fann_graph_execution import BoundedGraphExecution,ConstantSparseMultiply,csr_from_hexahedra


def lattice():
    nx,ny,nz=10,3,2
    points=np.array([(x*.03,y*.04,z*.05) for z in range(nz) for y in range(ny) for x in range(nx)],np.float32)
    def node(x,y,z):return (z*ny+y)*nx+x
    cells=[]
    for z in range(nz-1):
        for y in range(ny-1):
            for x in range(nx-1):
                cells.append([node(x,y,z),node(x+1,y,z),node(x+1,y+1,z),node(x,y+1,z),
                              node(x,y,z+1),node(x+1,y,z+1),node(x+1,y+1,z+1),node(x,y+1,z+1)])
    target=np.random.default_rng(8).normal(size=(len(points),6)).astype(np.float32)
    target[:,3]=target[:,3]*100+1e5;target[:,4]+=290;target[:,5]=target[:,5]*.01+1
    return {'coordinates':points,'hexahedra':np.array(cells),'original_node_ids':np.arange(len(points)),
            'wall_distance':np.linspace(0,.03,len(points),dtype=np.float32),
            'operating_conditions':np.array([900,1e5],np.float32),'targets':target}


class BoundedGraphTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.sample=lattice();self.norm=Normalizer([self.sample]);torch.set_num_threads(2)
    def executor(self,name):
        return BoundedGraphExecution(name,{'graph_owner_nodes':4,'graph_edge_chunk':7,'graph_cache_mb':1},self.root)
    def test_cpu_topology_matches_original_edges(self):
        graph=self.norm.graph(self.sample,'cpu');adj=csr_from_hexahedra(self.sample['hexahedra'],len(self.sample['coordinates']))
        coo=adj.tocoo();actual=set(zip(coo.col,coo.row));expected=set(map(tuple,graph.edge_index.T.numpy()))
        self.assertEqual(actual,expected)
    def test_owner_halo_predictions_losses_and_parameter_gradients(self):
        for name in ['GCN','Graph_Transformer']:
            with self.subTest(model=name):
                torch.manual_seed(12);dense=FannAdapter(name,[self.sample],self.norm)
                bounded=copy.deepcopy(dense);executor=self.executor(name)
                dense.train();bounded.train()
                prediction=dense(self.norm.graph(self.sample,'cpu'))
                loss=(prediction-self.norm.target(self.sample,'cpu')).square().mean();loss.backward()
                partition_loss=executor.backward_sample(bounded,self.norm,self.sample,'cpu')
                self.assertAlmostEqual(partition_loss,float(loss),places=5)
                for (key,a),(_,b) in zip(dense.named_parameters(),bounded.named_parameters()):
                    self.assertEqual(a.grad is None,b.grad is None,key)
                    if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=3e-4,atol=3e-6,msg=key)
                dense.eval();bounded.eval()
                with torch.no_grad():
                    expected=dense(self.norm.graph(self.sample,'cpu'))
                    actual=executor.predict_cpu(bounded,self.norm,self.sample,'cpu')
                torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-6)
                adjacency=executor.adjacency(self.sample)
                hops=bounded.net.processor_layers if name=='GCN' else bounded.net.num_layers
                owner_ids=[];smaller=False
                for start in range(0,len(self.sample['coordinates']),executor.owner_nodes):
                    part=executor.partition(adjacency,start,hops)
                    owner_ids.extend(part['nodes'][part['owners']].tolist())
                    smaller |= len(part['nodes'])<len(self.sample['coordinates'])
                self.assertEqual(owner_ids,list(range(len(self.sample['coordinates']))));self.assertTrue(smaller)
    def test_graph_unet_preserves_global_pooling_output_and_gradients(self):
        torch.manual_seed(19);dense=FannAdapter('Graph_UNet',[self.sample],self.norm);bounded=copy.deepcopy(dense)
        executor=self.executor('Graph_UNet');dense.train();bounded.train()
        expected=dense(self.norm.graph(self.sample,'cpu'));actual=executor.graph_unet(bounded,self.norm,self.sample,'cpu')
        torch.testing.assert_close(actual,expected,rtol=4e-4,atol=4e-6)
        target=self.norm.target(self.sample,'cpu')
        (expected-target).square().mean().backward();(actual-target).square().mean().backward()
        for (key,a),(_,b) in zip(dense.named_parameters(),bounded.named_parameters()):
            self.assertEqual(a.grad is None,b.grad is None,key)
            if a.grad is not None:torch.testing.assert_close(a.grad,b.grad,rtol=6e-4,atol=6e-6,msg=key)
    def test_normalization_preserves_existing_self_loops(self):
        from torch_geometric.nn.conv.gcn_conv import gcn_norm
        from tfai.benchmarks.fann_graph_execution import normalize_gcn
        matrix=sparse.csr_matrix(np.array([[3,2,0,0],[1,0,4,0],[0,2,0,0],[0,0,0,0]],np.float32))
        coo=matrix.tocoo();edges=torch.from_numpy(np.stack([coo.col,coo.row])).long()
        index,weight=gcn_norm(edges,torch.from_numpy(coo.data),4,improved=True)
        expected=torch.zeros(4,4);expected.index_put_((index[1],index[0]),weight,accumulate=True)
        actual=torch.from_numpy(normalize_gcn(matrix).toarray())
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)

    def test_sparse_adjoint_with_empty_rows_and_edge_chunk_boundaries(self):
        matrix=sparse.csr_matrix(np.array([[0,0,0,0],[1,0,2,0],[0,0,0,0],[0,3,0,4]],np.float32))
        values=torch.randn(4,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda x:ConstantSparseMultiply.apply(x,matrix,2),(values,)))
        actual=ConstantSparseMultiply.apply(values,matrix,2)
        expected=torch.as_tensor(matrix.toarray(),dtype=values.dtype)@values
        torch.testing.assert_close(actual,expected)
    def test_geo_streamed_slices_preserve_global_outputs_and_gradients(self):
        torch.manual_seed(7)
        dense=FannAdapter('GeoTransolver',[self.sample],self.norm);bounded=copy.deepcopy(dense)
        for module in bounded.net.model.modules():module.node_chunk_size=7
        a=self.norm.graph(self.sample,'cpu',with_edges=False);b=a.clone()
        a.x.requires_grad_();b.x.requires_grad_()
        expected=dense(a);actual=bounded(b)
        target=self.norm.target(self.sample,'cpu')
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=2e-6)
        (expected-target).square().mean().backward();(actual-target).square().mean().backward()
        torch.testing.assert_close(a.x.grad,b.x.grad,rtol=1e-3,atol=2e-6)
        for (key,p),(_,q) in zip(dense.named_parameters(),bounded.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,key)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-3,atol=2e-6,msg=key)

    @unittest.skipUnless(torch.cuda.device_count()>=2,'Two GPUs required for layer model parallelism')
    def test_four_models_layer_parallel_output_gradients_and_optimizer(self):
        from tfai.benchmarks.fann_model_parallel import configure
        for name in ['GCN','Graph_UNet','Graph_Transformer','GeoTransolver']:
            with self.subTest(model=name):
                torch.manual_seed(13)
                dense=FannAdapter(name,[self.sample],self.norm).to('cuda:0')
                parallel=copy.deepcopy(dense)
                if name=='GeoTransolver':
                    parallel.net.activation_offload=True
                    for module in parallel.net.model.modules():module.node_chunk_size=7
                if name=='Graph_UNet':parallel.graph_executor=self.executor(name)
                configure(parallel,'cuda:0',count=2)
                graph=self.norm.graph(self.sample,'cuda:0',with_edges=name!='GeoTransolver')
                expected=dense(graph)
                actual=(self.executor(name).graph_unet(parallel,self.norm,self.sample,'cuda:0')
                        if name=='Graph_UNet' else parallel(graph))
                torch.testing.assert_close(actual,expected,rtol=4e-4,atol=4e-6)
                target=self.norm.target(self.sample,'cuda:0')
                (expected-target).square().mean().backward();(actual-target).square().mean().backward()
                for (key,p),(_,q) in zip(dense.named_parameters(),parallel.named_parameters()):
                    self.assertEqual(p.grad is None,q.grad is None,key)
                    if p.grad is not None:torch.testing.assert_close(p.grad.cpu(),q.grad.cpu(),rtol=1e-3,atol=5e-6,msg=key)
                torch.nn.utils.clip_grad_norm_(parallel.parameters(),1.,error_if_nonfinite=True)
                optimizer=torch.optim.Adam(parallel.parameters(),lr=1e-4);optimizer.step()
                self.assertEqual({p.device.index for p in parallel.parameters()},{0,1})
                checkpoint=copy.deepcopy(parallel.state_dict())
                parallel.load_state_dict(checkpoint,strict=True)
                # A second full-case step also exercises Adam state on both GPUs.
                optimizer.zero_grad(set_to_none=True)
                output=(self.executor(name).graph_unet(parallel,self.norm,self.sample,'cuda:0')
                        if name=='Graph_UNet' else parallel(graph))
                (output-target).square().mean().backward();optimizer.step()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required for host activation offload')
    def test_geo_offload_preserves_outputs_parameter_and_input_gradients(self):
        torch.manual_seed(7)
        dense=FannAdapter('GeoTransolver',[self.sample],self.norm).cuda()
        offloaded=copy.deepcopy(dense);offloaded.net.activation_offload=True
        graph=self.norm.graph(self.sample,'cuda',with_edges=False)
        a=graph.clone();b=graph.clone();a.x.requires_grad_();b.x.requires_grad_()
        expected=dense(a);actual=offloaded(b)
        target=self.norm.target(self.sample,'cuda')
        (expected-target).square().mean().backward();(actual-target).square().mean().backward()
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(a.x.grad,b.x.grad,rtol=1e-4,atol=1e-6)
        self.assertEqual(set(dense.state_dict()),set(offloaded.state_dict()))
        for (key,p),(_,q) in zip(dense.named_parameters(),offloaded.named_parameters()):
            self.assertEqual(p.grad is None,q.grad is None,key)
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=1e-6,msg=key)
        offloaded.eval()
        with torch.inference_mode():
            torch.testing.assert_close(offloaded(graph),expected.detach(),rtol=1e-5,atol=1e-6)

    def test_owner_partition_can_interrupt_before_any_gradient(self):
        model=FannAdapter('GCN',[self.sample],self.norm)
        value=self.executor('GCN').backward_sample(model,self.norm,self.sample,'cpu',lambda:True)
        self.assertIsNone(value);self.assertTrue(all(p.grad is None for p in model.parameters()))

if __name__=='__main__':unittest.main()
