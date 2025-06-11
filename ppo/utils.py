"""
utils.py ― tiny helper utilities (GAE, RolloutBuffer, seeding)
"""

from __future__ import annotations
from dataclasses import dataclass
import torch


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