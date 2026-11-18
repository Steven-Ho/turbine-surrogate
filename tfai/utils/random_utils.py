import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch RNGs in one place."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)


def capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states for checkpoint resume."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Optional[Dict[str, Any]]) -> None:
    """Restore RNG states saved by ``capture_rng_state``."""
    if not state:
        return

    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)

    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)

    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)

    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        current_device_count = torch.cuda.device_count()
        if len(cuda_state) == current_device_count:
            torch.cuda.set_rng_state_all(cuda_state)
        else:
            for idx, device_state in enumerate(cuda_state[:current_device_count]):
                torch.cuda.set_rng_state(device_state, device=f"cuda:{idx}")
