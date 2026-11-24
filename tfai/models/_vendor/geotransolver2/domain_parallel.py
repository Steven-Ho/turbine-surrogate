"""Portable node-domain parallelism inspired by PhysicsNeMo ShardTensor.

Replicate small physical slices after SUM reductions, never train independent
patches. Uses only PyTorch distributed SUM/all_gather (no DTensor, TE, Warp,
custom CUDA kernels or reduce_scatter). Parameters retain their existing names.
"""
import torch
import torch.distributed as dist


class _Sum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, group):
        ctx.group = group
        with torch.inference_mode(False):
            result = value.contiguous().clone()
            dist.all_reduce(result, group=group)
        return result

    @staticmethod
    def backward(ctx, grad):
        result = grad.contiguous().clone()
        dist.all_reduce(result, group=ctx.group)
        return result, None


def domain_sum(module, value):
    group = getattr(module, 'domain_group', None)
    return value if group is None else _Sum.apply(value, group)


def bounds(nodes, rank, size):
    if nodes < size:
        raise ValueError('Domain-parallel case needs at least one node per domain rank')
    return nodes * rank // size, nodes * (rank + 1) // size


class Domain:
    def __init__(self, size):
        world = dist.get_world_size()
        if size < 1 or world % size:
            raise ValueError('World size must be divisible by domain size')
        self.size = size
        self.rank = dist.get_rank() % size
        self.data_rank = dist.get_rank() // size
        self.data_size = world // size
        self.group = None
        # All ranks create groups in identical order, including non-members.
        for start in range(0, world, size):
            group = dist.new_group(list(range(start, start + size)))
            if start <= dist.get_rank() < start + size:
                self.group = group
        self.full_xyz = None

    def attach(self, adapter):
        from .context_projector import ContextProjector, GeometricFeatureProcessor
        from .gale import GALE
        for name, module in adapter.model.named_modules():
            if isinstance(module, (ContextProjector, GALE)) and 'global_tokenizer' not in name:
                module.domain_group = self.group
            if isinstance(module, GeometricFeatureProcessor):
                module.domain = self
        adapter.domain = self

    def select(self, value):
        start, end = bounds(len(value), self.rank, self.size)
        return value[start:end]

    def gather(self, value):
        # Only used without gradients during validation/inference.
        if torch.is_grad_enabled():
            raise ValueError('Domain gather is inference-only')
        counts=[bounds(len(self.full_xyz),i,self.size) for i in range(self.size)]
        counts=[end-start for start,end in counts]
        # Distributed backends may complete writes on a worker thread outside
        # the caller's inference-mode context. Allocate normal output tensors.
        with torch.inference_mode(False):
            padded=torch.nn.functional.pad(value.clone(),(0,0,0,max(counts)-len(value))).contiguous()
            parts=[torch.empty_like(padded) for _ in counts]
            dist.all_gather(parts,padded,group=self.group)
            return torch.cat([part[:n] for part,n in zip(parts,counts)])



def neighborhood(processor, query_points, key_features):
    """Use the existing whole-geometry cache, transferring only owned rows.

    Coordinates are fixed non-differentiable inputs. Replicating them preserves
    cross-boundary neighbours without a feature halo exchange in this version.
    """
    domain = processor.domain
    points = domain.full_xyz
    if points is None or points.requires_grad:
        raise ValueError('Domain neighborhood requires fixed full-case coordinates')
    bq = processor.bq_warp
    cached = getattr(bq, '_domain_cached', None)
    if cached is not None and torch.equal(points, cached[0]):
        ids, valid = cached[1:]
    else:
        start, end = bounds(len(points), domain.rank, domain.size)
        if bq.backend == 'kdtree':
            raw = bq._spatial_raw(points, points)[start:end]
            signed = torch.as_tensor(raw.astype('int64'), device=points.device)
            ids, valid = signed.clamp_min(0), signed >= 0
        else:
            ids, valid = bq._dense_indices(points, points[start:end])
        bq._domain_cached = (points.detach().clone(), ids, valid)
    chunk = getattr(processor, 'node_chunk_size', 32768) or 32768
    from .memory import recompute
    parts = []
    for start in range(0, len(ids), chunk):
        index, mask = ids[start:start+chunk], valid[start:start+chunk]
        neighbors = points[index] * mask[..., None].to(points.dtype)
        if neighbors.shape[1] < bq.neighbors_in_radius:
            neighbors = torch.nn.functional.pad(neighbors, (0, 0, 0, bq.neighbors_in_radius-neighbors.shape[1]))
        parts.append(recompute(lambda values: torch.tanh(processor.mlp(values)), neighbors.reshape(1, len(index), -1)))
    return torch.cat(parts, dim=1)
