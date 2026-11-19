"""Rotor37 graph adapters around pinned official Transolver implementations.

Input: existing normalized [x,y,z,Omega,P] graph features. No target fields.
Output: normalized pressure at all input nodes, in their original order.
"""
import torch
from torch import nn
from ._vendor.transolver3.Transolver_chunk_opt_matrix_mul import Model as OfficialTransolver3

SCALES={"small":dict(layers=3,hidden=64,heads=4,slices=32),
        "base":dict(layers=4,hidden=128,heads=8,slices=64),
        "large":dict(layers=5,hidden=256,heads=8,slices=64)}


def graph_features(data, input_dim):
    x=getattr(data,"x",None)
    if not isinstance(x,torch.Tensor) or x.ndim!=2 or x.shape[1]!=input_dim:
        raise ValueError(f"Expected node features [N,{input_dim}]")
    batch=getattr(data,"batch",None)
    if batch is None:
        yield x
    else:
        # PyG Batch nodes are contiguous by graph, preserving output order.
        for graph_id in range(int(batch.max().item())+1):
            yield x[batch==graph_id]


class Transolver3(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        if input_dim<5: raise ValueError("Expected xyz, optional nodal features and two global conditions")
        if config.get("training",{}).get("graph_use_partition",False):
            raise ValueError("Use Transolver3 global slice chunking, not independent graph partitions")
        self.input_dim=input_dim
        s=SCALES[config["model"]["scale"]]
        self.chunk_size=int(config["model"].get("transolver_chunk_size",8192))
        self.model=OfficialTransolver3(space_dim=3,fun_dim=input_dim-3,out_dim=output_dim,
            n_layers=s["layers"],n_hidden=s["hidden"],n_head=s["heads"],slice_num=s["slices"],
            mlp_ratio=2,dropout=0.,unified_pos=False)

    def forward(self,data):
        outputs=[]
        for x in graph_features(data,self.input_dim):
            chunks=[part.unsqueeze(0) for part in x.split(self.chunk_size)]
            # Every chunk contributes to the same global physical state per layer.
            result=self.model(chunks,use_checkpoint=self.training,input_list=True)
            outputs.append(torch.cat(result,dim=1).squeeze(0))
        return torch.cat(outputs,dim=0)


class GeoTransolver(nn.Module):
    def __init__(self, config, input_dim, output_dim):
        super().__init__()
        backend=config["model"].get("geotransolver_backend","physicsnemo")
        if backend == "portable_torch":
            from ._vendor.geotransolver2 import GeoTransolver as OfficialGeoTransolver
        elif backend == "physicsnemo":
            from importlib.metadata import version
            if version("nvidia-physicsnemo")!="2.0.0":
                raise RuntimeError("This adapter is validated against nvidia-physicsnemo==2.0.0")
            from physicsnemo.experimental.models.geotransolver import GeoTransolver as OfficialGeoTransolver
        else:
            raise ValueError("geotransolver_backend must be physicsnemo or portable_torch")
        if input_dim<5: raise ValueError("Expected xyz, optional nodal features and two global conditions")
        if config.get("training",{}).get("graph_use_partition",False):
            raise ValueError("GeoTransolver requires whole-case geometry/global context")
        self.input_dim=input_dim
        s=SCALES[config["model"]["scale"]]
        self.model=OfficialGeoTransolver(functional_dim=input_dim-2,out_dim=output_dim,
            geometry_dim=3,global_dim=2,n_layers=s["layers"],n_hidden=s["hidden"],
            n_head=s["heads"],slice_num=s["slices"],mlp_ratio=2,dropout=0.,use_te=False,
            include_local_features=True,radii=[0.05,0.25],neighbors_in_radius=[8,32],n_hidden_local=16)

    def forward(self,data):
        outputs=[]
        for x in graph_features(data,self.input_dim):
            xyz=x[:,:3].unsqueeze(0)
            conditions=x[:1,-2:].unsqueeze(0)
            if not torch.allclose(x[:,-2:],x[:1,-2:].expand(x.shape[0],-1)):
                raise ValueError("Omega/P must be constant across nodes within a case")
            outputs.append(self.model(x[:,:-2].unsqueeze(0),local_positions=xyz,
                global_embedding=conditions,geometry=xyz).squeeze(0))
        return torch.cat(outputs,dim=0)
