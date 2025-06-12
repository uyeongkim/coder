# =============================================================
# utils.py  ──  tiny helper utilities for SH-PPO
#   * reproducible seeding across Python/NumPy/PyTorch (+GPU)
#   * light RolloutBuffer with Generalised Advantage Estimation
#   * misc tensor helpers (pad_stack, move_to)
# =============================================================

from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Dict, Any

import sys
import torch

# =============================================================
# Extra: Sanity check for NaN/infs in tensors
# =============================================================

def check_for_nan_and_abort(tensor: torch.Tensor, name: str, logger, context: str = ""):
    """
    Checks a tensor for NaNs or infinite values. Logs an error and aborts if any are found.
    """
    if not torch.isfinite(tensor).all():
        logger.error(f"Detected non-finite values in {name}. Context: {context}")
        sys.exit(1)

# ═════════════════════════════════════════════════════════════
# 1.  Global seeding
# ═════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Seed Python RNG, NumPy, PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ═════════════════════════════════════════════════════════════
# 2.  Rollout buffer (supports single‑turn or multi‑turn episodes)
# ═════════════════════════════════════════════════════════════
@dataclass
class Trajectory:
    state:  torch.Tensor          # observation embedding
    latent: torch.Tensor          # per-agent latent/hidden state
    action: torch.Tensor
    logp:   torch.Tensor
    reward: torch.Tensor | float
    value:  torch.Tensor

    # Keep trainer’s t.obs accesses working
    @property
    def obs(self) -> torch.Tensor:
        return self.state

class RolloutBuffer:
    """List‑based storage → convenient for variable‑length episodes."""

    def __init__(self, gamma: float, lam: float):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    # ----------------------------------------------------------
    def reset(self):
        self.traj: List[Trajectory] = []
        self.returns: torch.Tensor | None = None
        self.advs: torch.Tensor | None = None

    # ----------------------------------------------------------
    def add(self, tr: Trajectory):
        self.traj.append(tr)

    # ----------------------------------------------------------
    def compute_gae(self, last_value: torch.Tensor | float = 0.0):
        """Standard GAE‑λ over stored trajectory."""
        rewards = [t.reward for t in self.traj]
        values = [t.value for t in self.traj] + [torch.as_tensor(last_value)]
        advs, rets = [], []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advs.insert(0, gae)
            rets.insert(0, gae + values[t])
        self.advs = torch.stack([torch.as_tensor(a, dtype=torch.float32) for a in advs])
        self.returns = torch.stack([torch.as_tensor(r, dtype=torch.float32) for r in rets])
        
    def compute_returns_advantages(self, last_value: torch.Tensor | float = 0.0):
        """Alias expected by trainer.py; just calls `compute_gae`."""
        return self.compute_gae(last_value)

    @property
    def trajs(self) -> List[Trajectory]:
        """Alias so trainer can iterate over `self.buf.trajs`."""
        return self.traj

# ═════════════════════════════════════════════════════════════
# 3.  Small tensor helpers
# ═════════════════════════════════════════════════════════════

def pad_and_stack(seq: Sequence[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
    """Pad 1‑D tensors to max length then stack [N,L_max]."""
    if len(seq) == 0:
        return torch.empty(0)
    L = max(t.size(0) for t in seq)
    padded = []
    for t in seq:
        if t.size(0) < L:
            pad = torch.full((L - t.size(0),), pad_val, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad])
        padded.append(t)
    return torch.stack(padded)


def move_to(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors inside nested dict/list to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to(v, device) for v in obj)
    return obj
