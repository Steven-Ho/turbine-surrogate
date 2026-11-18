# ignore_header_test
# ruff: noqa: E402

r"""
Transolver model and building blocks for physics-informed neural operator learning.

This module provides the main Transolver model class along with its internal
building blocks (MLP, Transolver_block) for solving PDEs on structured and
unstructured meshes.

This code was modified from https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .mlp_layers import Mlp

class _TransolverMlp(Mlp):
    """Mlp subclass with state dict compatibility for legacy Transolver checkpoints.

    This class provides backward compatibility for loading checkpoints saved with
    the old Transolver MLP class, which used different attribute names:
    - Old: `linear_pre`, `linear_post`, `linears`
    - New: `layers` (Sequential)

    The mapping handles the common case where `n_layers=0` (no hidden layers with
    residual connections), which was the typical usage pattern in Transolver.
    """

    # Mapping from old checkpoint keys to new Mlp keys
    # This assumes the typical usage: n_layers=0, which means just linear_pre -> act -> linear_post
    # Old structure: linear_pre (input->hidden), linear_post (hidden->output)
    # New structure: layers.0 (input->hidden), layers.1 (activation), layers.2 (hidden->output)
    _OLD_TO_NEW_KEYS = {
        "linear_pre.weight": "layers.0.weight",
        "linear_pre.bias": "layers.0.bias",
        "linear_post.weight": "layers.2.weight",
        "linear_post.bias": "layers.2.bias",
    }

    _NEW_TO_OLD_KEYS = {v: k for k, v in _OLD_TO_NEW_KEYS.items()}

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list,
        unexpected_keys: list,
        assign: bool = False,
    ):
        """Load state dict with automatic key remapping for legacy checkpoints.

        This hook is called by PyTorch for each module during load_state_dict().
        We intercept it to remap old-style keys (linear_pre, linear_post) to
        new-style keys (layers.0, layers.2) before the actual loading.
        """
        # Check for old-style keys with this module's prefix
        # e.g., prefix="preprocess." -> look for "preprocess.linear_pre.weight"
        for old_suffix, new_suffix in self._OLD_TO_NEW_KEYS.items():
            old_key = prefix + old_suffix
            new_key = prefix + new_suffix
            if old_key in state_dict and new_key not in state_dict:
                # Remap old key to new key
                state_dict[new_key] = state_dict.pop(old_key)

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            assign,
        )
