# =============================================================
# utils.py  ──  Complete utilities for SHPPO
#   * Enhanced trajectory handling with latent variables
#   * Improved rollout buffer with GAE
#   * Rich logging utilities
#   * Tensor manipulation helpers
#   * Reproducible seeding
# =============================================================

from __future__ import annotations

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Sequence, Dict, Any, Optional, Union

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress
from rich.text import Text

# ═════════════════════════════════════════════════════════════
# 1. Global seeding
# ═════════════════════════════════════════════════════════════

def set_seed(seed: int):
    """Seed Python RNG, NumPy, PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Additional CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ═════════════════════════════════════════════════════════════
# 2. Enhanced Trajectory for SHPPO
# ═════════════════════════════════════════════════════════════

@dataclass
class Trajectory:
    """Enhanced trajectory with latent variables and SHPPO-specific data"""
    
    # Core RL data
    state: torch.Tensor           # observation embedding
    action: torch.Tensor          # selected action
    logp: torch.Tensor           # log probability of action
    reward: torch.Tensor | float  # reward received
    value: torch.Tensor          # value estimate
    
    # SHPPO-specific latent data
    latent: torch.Tensor         # sampled latent variable
    mu: Optional[torch.Tensor] = None      # latent distribution mean
    sigma: Optional[torch.Tensor] = None   # latent distribution std
    
    # Additional context
    obs_emb: Optional[torch.Tensor] = None  # raw observation embedding
    hidden_state: Optional[torch.Tensor] = None  # RNN hidden state
    
    # GAE computed values (filled later)
    returns: Optional[torch.Tensor] = None
    advs: Optional[torch.Tensor] = None
    
    # Keep trainer's backward compatibility
    @property
    def obs(self) -> torch.Tensor:
        """Backward compatibility alias"""
        return self.state
    
    def to_device(self, device: torch.device) -> 'Trajectory':
        """Move trajectory to device"""
        new_traj = Trajectory(
            state=self.state.to(device),
            action=self.action.to(device),
            logp=self.logp.to(device),
            reward=self.reward.to(device) if torch.is_tensor(self.reward) else torch.tensor(self.reward, device=device),
            value=self.value.to(device),
            latent=self.latent.to(device)
        )
        
        # Move optional tensors
        if self.mu is not None:
            new_traj.mu = self.mu.to(device)
        if self.sigma is not None:
            new_traj.sigma = self.sigma.to(device)
        if self.obs_emb is not None:
            new_traj.obs_emb = self.obs_emb.to(device)
        if self.hidden_state is not None:
            new_traj.hidden_state = self.hidden_state.to(device)
        if self.returns is not None:
            new_traj.returns = self.returns.to(device)
        if self.advs is not None:
            new_traj.advs = self.advs.to(device)
            
        return new_traj
    
    def has_gae_computed(self) -> bool:
        """Check if GAE has been computed"""
        return self.returns is not None and self.advs is not None
    
    def get_latent_info(self) -> Dict[str, torch.Tensor]:
        """Get all latent-related information"""
        info = {'latent': self.latent}
        if self.mu is not None:
            info['mu'] = self.mu
        if self.sigma is not None:
            info['sigma'] = self.sigma
        return info

# ═════════════════════════════════════════════════════════════
# 3. Enhanced Rollout Buffer with GAE
# ═════════════════════════════════════════════════════════════

class RolloutBuffer:
    """Enhanced rollout buffer with GAE computation and SHPPO support"""

    def __init__(self, gamma: float, lam: float):
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self):
        """Reset buffer for new episode"""
        self.traj: List[Trajectory] = []
        self.returns: Optional[torch.Tensor] = None
        self.advs: Optional[torch.Tensor] = None
        self._gae_computed = False

    def add(self, tr: Trajectory):
        """Add trajectory to buffer"""
        self.traj.append(tr)
        self._gae_computed = False  # Mark GAE as needing recomputation

    def __len__(self) -> int:
        return len(self.traj)

    def compute_gae(self, last_value: torch.Tensor | float = 0.0):
        """
        Compute Generalized Advantage Estimation (GAE-λ)
        
        Args:
            last_value: Value of the final state (0 if terminal)
        """
        if len(self.traj) == 0:
            return
            
        # Extract rewards and values
        rewards = []
        values = []
        
        for t in self.traj:
            if torch.is_tensor(t.reward):
                rewards.append(t.reward.item())
            else:
                rewards.append(float(t.reward))
            values.append(t.value.item())
        
        # Add last value for GAE computation
        if torch.is_tensor(last_value):
            last_val = last_value.item()
        else:
            last_val = float(last_value)
        values.append(last_val)
        
        # GAE computation
        advantages = []
        returns = []
        gae = 0.0
        
        for t in reversed(range(len(rewards))):
            # TD error
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            
            # GAE step
            gae = delta + self.gamma * self.lam * gae
            
            # Store in reverse order (will reverse back)
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        # Convert to tensors
        device = self.traj[0].state.device
        self.advs = torch.tensor(advantages, dtype=torch.float32, device=device)
        self.returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Normalize advantages
        if len(advantages) > 1:
            mean_adv = self.advs.mean()
            std_adv = self.advs.std()
            if std_adv > 1e-8:
                self.advs = (self.advs - mean_adv) / (std_adv + 1e-8)
        
        # Assign to individual trajectories
        for i, traj in enumerate(self.traj):
            traj.returns = self.returns[i]
            traj.advs = self.advs[i]
        
        self._gae_computed = True

    def compute_returns_advantages(self, last_value: torch.Tensor | float = 0.0):
        """Alias for compute_gae for backward compatibility"""
        self.compute_gae(last_value)

    @property
    def trajs(self) -> List[Trajectory]:
        """Backward compatibility alias"""
        return self.traj
    
    def get_all_latent_data(self) -> Dict[str, List[torch.Tensor]]:
        """Extract all latent data from trajectories"""
        mu_list = []
        sigma_list = []
        latent_list = []
        
        for traj in self.traj:
            latent_list.append(traj.latent)
            if traj.mu is not None:
                mu_list.append(traj.mu)
            if traj.sigma is not None:
                sigma_list.append(traj.sigma)
        
        return {
            'mu': mu_list,
            'sigma': sigma_list,
            'latent': latent_list
        }
    
    def to_device(self, device: torch.device):
        """Move entire buffer to device"""
        self.traj = [traj.to_device(device) for traj in self.traj]
        if self.returns is not None:
            self.returns = self.returns.to(device)
        if self.advs is not None:
            self.advs = self.advs.to(device)
    
    def get_statistics(self) -> Dict[str, float]:
        """Get buffer statistics"""
        if len(self.traj) == 0:
            return {}
        
        rewards = [t.reward.item() if torch.is_tensor(t.reward) else t.reward for t in self.traj]
        values = [t.value.item() for t in self.traj]
        
        stats = {
            'length': len(self.traj),
            'total_reward': sum(rewards),
            'mean_reward': sum(rewards) / len(rewards),
            'mean_value': sum(values) / len(values),
            'max_reward': max(rewards),
            'min_reward': min(rewards)
        }
        
        if self._gae_computed:
            stats.update({
                'mean_advantage': self.advs.mean().item(),
                'std_advantage': self.advs.std().item(),
                'mean_return': self.returns.mean().item()
            })
        
        return stats

# ═════════════════════════════════════════════════════════════
# 4. Rich Logging Utilities
# ═════════════════════════════════════════════════════════════

class SHPPOLogger:
    """Rich console logger for SHPPO training"""
    
    def __init__(self):
        self.console = Console()
    
    def log_training_start(self, config: Any):
        """Log training start with configuration"""
        table = Table(title="🚀 SHPPO Training Configuration")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Add configuration items
        if hasattr(config, 'n_episodes'):
            table.add_row("Episodes", str(config.n_episodes))
        if hasattr(config, 'device'):
            table.add_row("Device", str(config.device))
        if hasattr(config, 'lr_actor'):
            table.add_row("Actor LR", f"{config.lr_actor:.2e}")
        if hasattr(config, 'lr_critic'):
            table.add_row("Critic LR", f"{config.lr_critic:.2e}")
        if hasattr(config, 'lr_latent'):
            table.add_row("Latent LR", f"{config.lr_latent:.2e}")
        if hasattr(config, 'lambda_entropy'):
            table.add_row("Lambda Entropy", str(config.lambda_entropy))
        if hasattr(config, 'lambda_distance'):
            table.add_row("Lambda Distance", str(config.lambda_distance))
        
        self.console.print(table)
    
    def log_episode_summary(self, episode: int, metrics: Dict[str, float]):
        """Log episode summary with metrics"""
        # Performance indicators
        reward = metrics.get('episode_reward', 0.0)
        pass_rate = metrics.get('pass_rate', 0.0)
        
        if pass_rate >= 0.8:
            status = "[bold green]🎯 Excellent[/bold green]"
            status_color = "green"
        elif pass_rate >= 0.5:
            status = "[yellow]⚡ Good[/yellow]"
            status_color = "yellow"
        elif pass_rate >= 0.1:
            status = "[orange3]🔄 Learning[/orange3]"
            status_color = "orange3"
        else:
            status = "[red]💪 Training[/red]"
            status_color = "red"
        
        # Create summary text
        summary = Text()
        summary.append(f"Episode {episode:4d} | ", style="bold")
        summary.append(f"Reward: {reward:6.3f} | ", style="cyan")
        summary.append(f"Pass: {pass_rate:5.3f} | ", style="magenta")
        summary.append(f"Status: ", style="dim")
        summary.append(status)
        
        self.console.print(summary)
    
    def log_loss_breakdown(self, losses: Dict[str, float]):
        """Log detailed loss breakdown"""
        table = Table(title="📊 Loss Breakdown")
        table.add_column("Loss Type", style="cyan")
        table.add_column("Value", style="magenta", justify="right")
        table.add_column("Category", style="green")
        
        # Categorize losses
        for loss_name, loss_value in losses.items():
            if 'actor' in loss_name:
                category = "Actor"
            elif 'critic' in loss_name:
                category = "Critic"
            elif 'latent' in loss_name:
                category = "Latent"
            elif 'inference' in loss_name:
                category = "Inference"
            elif 'llm' in loss_name:
                category = "LLM"
            else:
                category = "Other"
            
            table.add_row(
                loss_name.replace('_', ' ').title(),
                f"{loss_value:.6f}",
                category
            )
        
        self.console.print(table)
    
    def log_buffer_statistics(self, role: str, buffer_stats: Dict[str, float]):
        """Log rollout buffer statistics"""
        if not buffer_stats:
            return
            
        text = Text()
        text.append(f"{role.title()} Buffer: ", style="bold")
        text.append(f"Length={buffer_stats.get('length', 0):2d} | ", style="dim")
        text.append(f"Reward={buffer_stats.get('mean_reward', 0):.3f} | ", style="cyan")
        text.append(f"Value={buffer_stats.get('mean_value', 0):.3f}", style="magenta")
        
        if 'mean_advantage' in buffer_stats:
            text.append(f" | Adv={buffer_stats['mean_advantage']:.3f}", style="yellow")
        
        self.console.print(text)
    
    def log_checkpoint_saved(self, path: str, episode: int):
        """Log checkpoint save"""
        self.console.print(f"[blue]💾 Checkpoint saved at episode {episode}: {path}[/blue]")
    
    def log_error(self, message: str, exception: Optional[Exception] = None):
        """Log error message"""
        self.console.print(f"[red]❌ Error: {message}[/red]")
        if exception:
            self.console.print(f"[dim]{str(exception)}[/dim]")

# ═════════════════════════════════════════════════════════════
# 5. Tensor manipulation helpers
# ═════════════════════════════════════════════════════════════

def pad_and_stack(seq: Sequence[torch.Tensor], pad_val: int = 0) -> torch.Tensor:
    """Pad 1-D tensors to max length then stack [N, L_max]."""
    if len(seq) == 0:
        return torch.empty(0)
    
    L = max(t.size(0) for t in seq)
    padded = []
    
    for t in seq:
        if t.size(0) < L:
            pad_size = L - t.size(0)
            if t.dim() == 1:
                pad = torch.full((pad_size,), pad_val, dtype=t.dtype, device=t.device)
            else:
                pad_shape = (pad_size,) + t.shape[1:]
                pad = torch.full(pad_shape, pad_val, dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=0)
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

def stack_trajectories(trajectories: List[Trajectory]) -> Dict[str, torch.Tensor]:
    """Stack trajectory data into batched tensors"""
    if not trajectories:
        return {}
    
    batch_data = {}
    
    # Stack basic trajectory data
    batch_data['states'] = torch.stack([t.state for t in trajectories])
    batch_data['actions'] = torch.stack([t.action for t in trajectories])
    batch_data['logps'] = torch.stack([t.logp for t in trajectories])
    batch_data['values'] = torch.stack([t.value for t in trajectories])
    batch_data['latents'] = torch.stack([t.latent for t in trajectories])
    
    # Handle rewards (can be tensors or floats)
    rewards = []
    for t in trajectories:
        if torch.is_tensor(t.reward):
            rewards.append(t.reward)
        else:
            rewards.append(torch.tensor(t.reward, device=t.state.device))
    batch_data['rewards'] = torch.stack(rewards)
    
    # Stack optional data if available
    if all(t.mu is not None for t in trajectories):
        batch_data['mus'] = torch.stack([t.mu for t in trajectories])
    
    if all(t.sigma is not None for t in trajectories):
        batch_data['sigmas'] = torch.stack([t.sigma for t in trajectories])
    
    if all(t.obs_emb is not None for t in trajectories):
        batch_data['obs_embs'] = torch.stack([t.obs_emb for t in trajectories])
    
    if all(t.returns is not None for t in trajectories):
        batch_data['returns'] = torch.stack([t.returns for t in trajectories])
    
    if all(t.advs is not None for t in trajectories):
        batch_data['advantages'] = torch.stack([t.advs for t in trajectories])
    
    return batch_data

def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize advantages to have zero mean and unit variance"""
    if advantages.numel() <= 1:
        return advantages
    
    mean = advantages.mean()
    std = advantages.std()
    
    if std < eps:
        return advantages - mean
    
    return (advantages - mean) / (std + eps)

def compute_explained_variance(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Compute explained variance for value function evaluation
    
    explained_var = 1 - Var[y_true - y_pred] / Var[y_true]
    """
    if y_true.numel() <= 1:
        return 0.0
    
    var_y = torch.var(y_true)
    if var_y < 1e-8:
        return 0.0
    
    var_residual = torch.var(y_true - y_pred)
    explained_var = 1.0 - (var_residual / var_y)
    
    return explained_var.item()

# ═════════════════════════════════════════════════════════════
# 6. Data processing utilities
# ═════════════════════════════════════════════════════════════

def batch_trajectories_by_role(
    buffers: Dict[str, RolloutBuffer], 
    device: torch.device
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Batch trajectory data by role for efficient processing"""
    batched_data = {}
    
    for role, buffer in buffers.items():
        if len(buffer.traj) > 0:
            # Move trajectories to device first
            buffer.to_device(device)
            
            # Stack trajectory data
            batched_data[role] = stack_trajectories(buffer.traj)
    
    return batched_data

def create_mini_batches(
    data: Dict[str, torch.Tensor], 
    batch_size: int,
    shuffle: bool = True
) -> List[Dict[str, torch.Tensor]]:
    """Create mini-batches from trajectory data"""
    if not data:
        return []
    
    # Get total number of samples
    n_samples = len(next(iter(data.values())))
    
    # Create indices
    indices = torch.randperm(n_samples) if shuffle else torch.arange(n_samples)
    
    # Create mini-batches
    mini_batches = []
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        
        mini_batch = {
            key: tensor[batch_indices] 
            for key, tensor in data.items()
        }
        mini_batches.append(mini_batch)
    
    return mini_batches

# ═════════════════════════════════════════════════════════════
# 7. Metrics and evaluation utilities
# ═════════════════════════════════════════════════════════════

class MetricsTracker:
    """Track and analyze training metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics: Dict[str, List[float]] = {}
    
    def add_metric(self, name: str, value: float):
        """Add a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append(value)
        
        # Keep only recent values
        if len(self.metrics[name]) > self.window_size:
            self.metrics[name] = self.metrics[name][-self.window_size:]
    
    def get_recent_mean(self, name: str, n: Optional[int] = None) -> float:
        """Get mean of recent n values (or all if n is None)"""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        
        values = self.metrics[name]
        if n is not None:
            values = values[-n:]
        
        return sum(values) / len(values)
    
    def get_recent_std(self, name: str, n: Optional[int] = None) -> float:
        """Get standard deviation of recent n values"""
        if name not in self.metrics or len(self.metrics[name]) < 2:
            return 0.0
        
        values = self.metrics[name]
        if n is not None:
            values = values[-n:]
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        
        return variance ** 0.5
    
    def get_trend(self, name: str, n: int = 20) -> str:
        """Get trend direction for recent n values"""
        if name not in self.metrics or len(self.metrics[name]) < n:
            return "insufficient_data"
        
        values = self.metrics[name][-n:]
        first_half = values[:n//2]
        second_half = values[n//2:]
        
        mean_first = sum(first_half) / len(first_half)
        mean_second = sum(second_half) / len(second_half)
        
        if mean_second > mean_first * 1.05:
            return "improving"
        elif mean_second < mean_first * 0.95:
            return "declining"
        else:
            return "stable"
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'current': values[-1],
                    'mean': sum(values) / len(values),
                    'std': self.get_recent_std(name),
                    'min': min(values),
                    'max': max(values),
                    'trend': self.get_trend(name)
                }
        
        return summary

# ═════════════════════════════════════════════════════════════
# 8. Configuration validation
# ═════════════════════════════════════════════════════════════

def validate_shppo_config(config: Any) -> List[str]:
    """Validate SHPPO configuration and return list of issues"""
    issues = []
    
    # Check learning rates
    lr_fields = ['lr_actor', 'lr_critic', 'lr_latent', 'lr_inference']
    for field in lr_fields:
        if hasattr(config, field):
            lr = getattr(config, field)
            if lr <= 0 or lr > 1.0:
                issues.append(f"Learning rate {field} = {lr} should be in (0, 1]")
    
    # Check loss weights
    if hasattr(config, 'lambda_entropy') and config.lambda_entropy < 0:
        issues.append(f"lambda_entropy = {config.lambda_entropy} should be >= 0")
    
    if hasattr(config, 'lambda_distance') and config.lambda_distance < 0:
        issues.append(f"lambda_distance = {config.lambda_distance} should be >= 0")
    
    # Check PPO parameters
    if hasattr(config, 'clip_eps'):
        if config.clip_eps <= 0 or config.clip_eps > 1.0:
            issues.append(f"clip_eps = {config.clip_eps} should be in (0, 1]")
    
    if hasattr(config, 'gamma'):
        if config.gamma <= 0 or config.gamma > 1.0:
            issues.append(f"gamma = {config.gamma} should be in (0, 1]")
    
    if hasattr(config, 'lam'):
        if config.lam < 0 or config.lam > 1.0:
            issues.append(f"lam = {config.lam} should be in [0, 1]")
    
    # Check model dimensions
    if hasattr(config, 'model_config'):
        model_cfg = config.model_config
        if hasattr(model_cfg, 'latent_dim') and model_cfg.latent_dim <= 0:
            issues.append(f"latent_dim = {model_cfg.latent_dim} should be > 0")
    
    return issues

# ═════════════════════════════════════════════════════════════
# 9. Memory management utilities
# ═════════════════════════════════════════════════════════════

def clear_gpu_cache():
    """Clear GPU cache if using CUDA"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_memory_info() -> Dict[str, float]:
    """Get memory usage information"""
    info = {}
    
    if torch.cuda.is_available():
        # GPU memory
        gpu_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        gpu_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        gpu_max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        info.update({
            'gpu_allocated_gb': gpu_allocated,
            'gpu_reserved_gb': gpu_reserved,
            'gpu_max_allocated_gb': gpu_max_allocated
        })
    
    # CPU memory (approximate)
    import psutil
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1024**3  # GB
    info['cpu_memory_gb'] = cpu_memory
    
    return info

def log_memory_usage(console: Optional[Console] = None):
    """Log current memory usage"""
    if console is None:
        console = Console()
    
    memory_info = get_memory_info()
    
    text = Text("Memory Usage: ", style="bold")
    for key, value in memory_info.items():
        text.append(f"{key}: {value:.2f}GB ", style="cyan")
    
    console.print(text)

# ═════════════════════════════════════════════════════════════
# 10. Export utilities
# ═════════════════════════════════════════════════════════════

__all__ = [
    # Core classes
    'Trajectory',
    'RolloutBuffer',
    'SHPPOLogger',
    'MetricsTracker',
    
    # Utility functions
    'set_seed',
    'pad_and_stack',
    'move_to',
    'stack_trajectories',
    'normalize_advantages',
    'compute_explained_variance',
    'batch_trajectories_by_role',
    'create_mini_batches',
    'validate_shppo_config',
    'clear_gpu_cache',
    'get_memory_info',
    'log_memory_usage'
]