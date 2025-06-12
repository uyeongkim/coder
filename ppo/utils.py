"""
utils.py ― tiny helper utilities (GAE, RolloutBuffer, seeding)
"""

from __future__ import annotations
from dataclasses import dataclass
import torch
import sys


# ──────────────────────────────────────────────────────────────────────────
# Global seeding helper
# ──────────────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    """
    Seed Python `random`, NumPy, and (multi-)GPU PyTorch RNGs.
    """
    import random, numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────
# Rollout buffer for PPO
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class RolloutBuffer:
    """
    Very lightweight list-based rollout buffer—perfectly fine for one-step episodes.
    Stores everything needed for PPO GAE advantage computation.
    """

    gamma: float
    lam: float

    # These lists are initialised in `reset`
    states: list | None = None
    actions: list | None = None
    logprobs: list | None = None
    rewards: list | None = None
    values: list | None = None
    returns: torch.Tensor | None = None
    advantages: torch.Tensor | None = None
    prompts: list[str] | None = None

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self.reset()

    # ------------------------------------------------------------------ #
    def add(self, state, action, logprob, reward, value):
        """
        Append one transition (we treat the whole episode as a single transition
        for whole-code PPO).
        """
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)

    # ------------------------------------------------------------------ #
    def compute_returns_advantages(self, last_value: torch.Tensor | None = None):
        """
        Classic Generalised Advantage Estimation (GAE-λ).
        Because we use single-step episodes, γ = λ = 1 and
        `returns = rewards`, `advantages = rewards − values`
        — but we keep the full formula for completeness.
        """
        rewards = self.rewards
        values = self.values + ([last_value] if last_value is not None else [0.0])

        gae = 0.0
        returns, advs = [], []

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * gae
            advs.insert(0, gae)
            returns.insert(0, gae + values[t])

        self.advantages = torch.stack(advs).float()
        self.returns = torch.stack(returns).float()

    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear stored trajectories."""
        self.states, self.actions, self.logprobs = [], [], []
        self.rewards, self.values = [], []
        self.prompts = []
        self.returns, self.advantages = None, None
        
        
# ──────────────────────────────────────────────────────────────────────────
# Else
# ──────────────────────────────────────────────────────────────────────────

def check_for_nan_and_abort(tensor, name, logger, step_info="",):
    """Check tensor for NaN and abort if found"""
    if torch.isnan(tensor).any():
        logger.error((
            f"❌ NaN detected in {name} at {step_info}\n"
            f"   Tensor shape: {tensor.shape}\n"
            f"   Tensor stats: min={tensor.min()}, max={tensor.max()}, mean={tensor.mean()}\n"
            f"   NaN count: {torch.isnan(tensor).sum()}\n"
            f"   Inf count: {torch.isinf(tensor).sum()}\n"
            "🚨 ABORTING PROCESS DUE TO NaN"
        ))
        sys.exit(1)
        
# Simple tokenization cache for repeated prompts
class SimpleCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.max_size = max_size
    
    def get_or_compute(self, key, compute_fn):
        if key in self.cache:
            return self.cache[key]
        
        result = compute_fn()
        
        # Simple cache management
        if len(self.cache) >= self.max_size:
            # Remove first item (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[key] = result
        return result