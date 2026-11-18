# Radius-query semantics adapted from PhysicsNeMo 2.0.0 (Apache-2.0).
# Original implementation: Copyright NVIDIA CORPORATION & AFFILIATES.
# See LICENSE.txt and UPSTREAM.json for source and license information.
"""Spatial-index radius query with content-addressed persistence and Torch fallback.

PhysicsNeMo explicitly does not guarantee identical capped-neighbor ordering
between Warp and Torch backends. This port uses the documented Torch baseline:
nearest points, radius-inclusive boundary, zero padding. Near ties and radius
boundaries use the legacy Torch chunk to preserve its selection ordering. The model architecture
and state-dict keys are unchanged. Backend/cache settings are recorded by the
optimized training launcher.
"""
import os
import time
import numpy as np
import torch
from .spatial_neighbors import NeighborIndexCache, geometry_key, tree_candidates
from torch import nn


class BQWarp(nn.Module):
    # Keep the upstream attribute/class name for state-dict/module layout parity.
    def __init__(self, radius=.25, neighbors_in_radius=10, query_chunk_size=256, backend=None):
        super().__init__()
        if neighbors_in_radius is None or neighbors_in_radius <= 0:
            raise ValueError('This GeoTransolver adapter requires a positive fixed neighbor count')
        if radius < 0 or query_chunk_size <= 0:
            raise ValueError('Invalid radius or query chunk size')
        self.radius = radius
        self.neighbors_in_radius = neighbors_in_radius
        self.query_chunk_size = query_chunk_size
        self._cached = None
        self.cache_hits = 0
        self.cache_misses = 0
        self.backend = backend or os.environ.get('TURBINE_GEO_QUERY_BACKEND', 'kdtree')
        if self.backend not in ('kdtree', 'torch'):
            raise ValueError('TURBINE_GEO_QUERY_BACKEND must be kdtree or torch')
        cache_dir = os.environ.get('TURBINE_GEO_CACHE_DIR')
        if cache_dir is None and os.environ.get('TFAI_PROJECT_ROOT'):
            cache_dir = os.path.join(os.environ['TFAI_PROJECT_ROOT'], 'runs', 'geo_neighbor_cache_kdtree_v1')
        self.index_cache = NeighborIndexCache(cache_dir,
                                              os.environ.get('TURBINE_GEO_MEMORY_CACHE', '32'))
        self.device_cache_hits = 0
        self.memory_cache_hits = 0
        self.disk_cache_hits = 0
        self.tree_builds = 0
        self.ambiguity_rows = 0
        self.reference_chunks = 0
        self.last_build_seconds = 0.

    def _apply(self, fn):
        self._cached = None
        return super()._apply(fn)

    def _indices(self, points, queries):
        cached = self._cached
        if cached is not None:
            old_points, old_queries, indices, valid = cached
            if (points.shape == old_points.shape and queries.shape == old_queries.shape
                    and points.device == old_points.device and points.dtype == old_points.dtype
                    and queries.device == old_queries.device and queries.dtype == old_queries.dtype
                    and torch.equal(points.detach(), old_points)
                    and torch.equal(queries.detach(), old_queries)):
                self.cache_hits += 1
                self.device_cache_hits += 1
                return indices, valid
        if self.backend == 'torch':
            self.cache_misses += 1
            indices, valid = self._dense_indices(points, queries)
        else:
            indices, valid = self._spatial_indices(points, queries)
        self._cached = (points.detach().clone(), queries.detach().clone(), indices, valid)
        return indices, valid

    def _dense_indices(self, points, queries):
        k = min(self.neighbors_in_radius, points.shape[0])
        all_indices, all_valid = [], []
        with torch.no_grad():
            for chunk in queries.detach().split(self.query_chunk_size):
                distances = torch.cdist(points.detach(), chunk, p=2.,
                                        compute_mode='donot_use_mm_for_euclid_dist')
                values, indices = torch.topk(distances, k=k, dim=0, largest=False)
                valid = values <= self.radius
                all_indices.append(torch.where(valid, indices, 0).T.contiguous())
                all_valid.append(valid.T.contiguous())
        return torch.cat(all_indices, dim=0), torch.cat(all_valid, dim=0)

    def _spatial_indices(self, points, queries):
        if points.dtype not in (torch.float32, torch.float64) or queries.dtype != points.dtype:
            raise ValueError('Spatial query requires matching float32/float64 coordinates; use backend=torch otherwise')
        with torch.no_grad():
            p = np.ascontiguousarray(points.detach().cpu().numpy())
            q = np.ascontiguousarray(queries.detach().cpu().numpy())
            device_name = torch.cuda.get_device_name(points.device) if points.is_cuda else 'cpu'
            context = {'torch': torch.__version__, 'device': points.device.type,
                       'device_name': device_name, 'chunk': self.query_chunk_size}
            key = geometry_key(p, q, self.radius, self.neighbors_in_radius, context)
            k = min(self.neighbors_in_radius, len(p))
            raw, source = self.index_cache.get(key, (len(q), k), len(p))
            if raw is None:
                self.cache_misses += 1
                self.tree_builds += 1
                start = time.perf_counter()
                raw, ambiguous = tree_candidates(p, q, self.radius, self.neighbors_in_radius)
                self.ambiguity_rows += int(ambiguous.sum())
                # Recompute whole ORIGINAL query chunks at ties/rounding boundaries.
                # This preserves legacy topk ordering/precision rather than guessing.
                blocks = np.unique(np.flatnonzero(ambiguous) // self.query_chunk_size)
                self.reference_chunks += len(blocks)
                for block in blocks:
                    lo = int(block) * self.query_chunk_size
                    hi = min(lo + self.query_chunk_size, len(q))
                    ids, valid = self._dense_indices(points, queries[lo:hi])
                    ids = ids.cpu().numpy()
                    valid = valid.cpu().numpy()
                    raw[lo:hi] = np.where(valid, ids, -1).astype(np.int32)
                self.index_cache.put(key, raw)
                self.last_build_seconds = time.perf_counter() - start
            else:
                self.cache_hits += 1
                if source == 'memory': self.memory_cache_hits += 1
                if source == 'disk': self.disk_cache_hits += 1
            signed = torch.as_tensor(raw.astype(np.int64), device=points.device)
            valid = signed >= 0
            return signed.clamp_min(0), valid

    def forward(self, x, p_grid, reverse_mapping=True):
        if x.shape[0] != 1 or p_grid.shape[0] != 1:
            raise ValueError('GeoTransolver radius query requires batch size 1')
        if x.ndim != 3 or p_grid.ndim not in (3, 4, 5) or x.shape[-1] != 3 or p_grid.shape[-1] != 3:
            raise ValueError('Expected coordinate tensors [1,N,3] or structured query grids')
        p_grid = p_grid.reshape(1, -1, 3)
        points, queries = (x[0], p_grid[0]) if reverse_mapping else (p_grid[0], x[0])
        if points.shape[0] == 0 or queries.shape[0] == 0:
            raise ValueError('Empty point cloud is unsupported')
        indices, valid = self._indices(points, queries)
        neighbors = points[indices] * valid.unsqueeze(-1).to(points.dtype)
        if neighbors.shape[1] < self.neighbors_in_radius:
            neighbors = torch.nn.functional.pad(neighbors, (0, 0, 0, self.neighbors_in_radius - neighbors.shape[1]))
        return indices.unsqueeze(0), neighbors.unsqueeze(0)
