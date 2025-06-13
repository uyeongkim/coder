from __future__ import annotations
import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich.logging import RichHandler
import wandb

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 기존 yy_env.py 사용 (완성된 환경)
from yy_env import (
    EnvConfig, create_env
)
from shppo.model import (
    SHPPOModelConfig, RoleConfig, EnhancedSharedLLM, 
    build_enhanced_actor_and_critic, SHPPOLossComputer
)
from shppo.utils import RolloutBuffer, Trajectory, set_seed, move_to

# Type hints for env.py compatibility
type_obs = Dict[str, Any]
type_act = Dict[str, str]

@dataclass
class SHPPOConfig:
    # Environment
    env_config: EnvConfig = field(default_factory=lambda: EnvConfig(
        max_problems=100,
        batch_size=4
    ))
    
    # Model
    model_config: SHPPOModelConfig = field(default_factory=lambda: SHPPOModelConfig(
        base_model_name="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        lora_r=8,
        lora_alpha=16,
        latent_dim=4,
        hete_layer_input_dim=128,
        hete_layer_output_dim=32,
        mlp_hidden_dim=64
    ))
    
    # Roles
    roles: Dict[str, RoleConfig] = field(default_factory=lambda: {
        "planner": RoleConfig("planner", obs_embed_dim=128, n_action_templates=4),
        "coder": RoleConfig("coder", obs_embed_dim=128, n_action_templates=4),
        "debugger": RoleConfig("debugger", obs_embed_dim=128, n_action_templates=4),
    })
    
    # Training
    n_episodes: int = 100
    n_epochs: int = 4
    mini_batch_size: int = 2
    
    # Separate Learning Rates for Each Component
    lr_actor: float = 3e-4
    lr_critic: float = 1e-3
    lr_latent: float = 5e-4
    lr_inference: float = 1e-3
    lr_llm: float = 1e-5
    lr_projection: float = 1e-3
    
    # PPO Parameters
    gamma: float = 0.95
    lam: float = 0.9
    clip_eps: float = 0.2
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    
    # Loss Weights (Equation 10)
    lambda_entropy: float = 0.01
    lambda_distance: float = 0.1
    
    # Training Control - REMOVED update_llm choice
    use_enhanced_generation: bool = True
    
    # Logging
    log_interval: int = 5
    eval_interval: int = 20
    save_interval: int = 50
    checkpoint_dir: str = "/data/checkpoints"
    wandb_project: str = "shppo"
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

class CompleteSHPPOTrainer:
    """Complete SHPPO Trainer with all loss functions and separate optimizers"""
    
    def __init__(self, config: SHPPOConfig):
        self.cfg = config
        set_seed(self.cfg.seed)
        
        # Rich console setup
        self.console = Console()
        
        # Training mode flag (like PyTorch modules)
        self.training = True
        
        # Setup
        self._setup_logging()
        
        # Create environment using yy_env
        self.env = create_env("simple_simple", self.cfg.env_config)
        
        self._build_models()
        self._setup_optimizers()
        self._setup_loss_computer()
        
        # Buffers and states
        self.buffers = {role: RolloutBuffer(self.cfg.gamma, self.cfg.lam) 
                       for role in self.cfg.roles.keys()}
        
        device = torch.device(self.cfg.device)
        self.hidden_states = {
            role: torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                            device=device, dtype=torch.float32) 
            for role in self.cfg.roles.keys()
        }
        
        # Training state
        self.episode = 0
        self.step = 0
        self.best_reward = float('-inf')
        self.metrics = defaultdict(list)
        
        self.console.print(Panel(
            "[bold green]✅ Complete SHPPO Trainer Initialized[/bold green]\n"
            f"• Device: {self.cfg.device}\n"
            f"• Roles: {list(self.cfg.roles.keys())}\n"
            f"• Episodes: {self.cfg.n_episodes}\n"
            f"• LLM Training: Always Enabled\n"
            f"• Optimizers: {len(self.opt_actors) + 4}",
            title="🚀 SHPPO Setup Complete"
        ))
    
    def train_mode(self, mode: bool = True):
        """Set training mode for the trainer and all models"""
        self.training = mode
        for actor in self.actors.values():
            actor.train(mode)
        self.critic.train(mode)
        self.inference_net.train(mode)
        self.llm.model.train(mode)
        self.obs_projection.train(mode)
        return self
    
    def eval_mode(self):
        """Set evaluation mode"""
        return self.train_mode(False)
        
    def _setup_logging(self):
        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        
        # Rich handler for beautiful logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[
                RichHandler(console=Console(), rich_tracebacks=True),
                logging.FileHandler(f'{self.cfg.checkpoint_dir}/training.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # WandB setup
        wandb.init(
            project=self.cfg.wandb_project,
            name=self.cfg.wandb_run_name or f"complete-run-{int(time.time())}",
            config=self.cfg.__dict__,
            reinit=True
        )
        
        # Rich console
        if not hasattr(self, 'console'):
            self.console = Console()
        
    def _build_models(self):
        device = torch.device(self.cfg.device)
        
        # Enhanced LLM with proper generation
        self.llm = EnhancedSharedLLM(self.cfg.model_config, device)
        
        # Build enhanced actor-critic with scalable inference net
        self.actors, self.critic, self.inference_net = build_enhanced_actor_and_critic(
            self.cfg.model_config, self.cfg.roles, device
        )
        
        # Observation projection (LLM embeddings -> RL embeddings)
        self.obs_projection = nn.Linear(
            self.llm.model.config.hidden_size,
            self.cfg.model_config.hete_layer_input_dim
        ).to(device)
        
        self.console.print(f"[green]✅ Models built on {device}[/green]")
        
    def _setup_optimizers(self):
        """Setup separate optimizers for each component"""
        
        # 1. Actor optimizers (per role)
        self.opt_actors = {
            role: torch.optim.Adam(actor.parameters(), lr=self.cfg.lr_actor)
            for role, actor in self.actors.items()
        }
        
        # 2. Critic optimizer
        critic_params = list(self.critic.parameters()) + list(self.obs_projection.parameters())
        self.opt_critic = torch.optim.Adam(critic_params, lr=self.cfg.lr_critic)
        
        # 3. Latent network optimizer (separate from actors)
        latent_params = []
        for actor in self.actors.values():
            latent_params.extend(actor.enc.parameters())  # Encoder parameters
        self.opt_latent = torch.optim.Adam(latent_params, lr=self.cfg.lr_latent)
        
        # 4. Inference network optimizer
        self.opt_inference = torch.optim.Adam(
            self.inference_net.parameters(), lr=self.cfg.lr_inference
        )
        
        # 5. LLM optimizer - ALWAYS ENABLED
        self.llm.set_llm_training_mode(True)  # Always enable LLM training
        llm_params = self.llm.trainable_parameters()
        if llm_params:
            self.opt_llm = torch.optim.Adam(llm_params, lr=self.cfg.lr_llm)
            self.console.print("[green]✅ LLM LoRA parameters enabled for training[/green]")
        else:
            self.opt_llm = None
            self.console.print("[red]❌ No trainable LLM parameters found[/red]")
        
        self.console.print(f"[green]✅ Setup {len(self.opt_actors)} actor optimizers + 4 other optimizers[/green]")
        
    def _setup_loss_computer(self):
        """Setup loss computer with paper's parameters"""
        self.loss_computer = SHPPOLossComputer(
            lambda_e=self.cfg.lambda_entropy,
            lambda_d=self.cfg.lambda_distance
        )
    
    def _embed_observation(self, obs: type_obs) -> torch.Tensor:
        """Embed observation using enhanced LLM"""
        role = obs["role"]
        
        # Handle visible files format
        files_text = "\n".join([f"=== {fname} ===\n{content[:200]}..." 
                            for fname, content in obs["visible_files"].items()])
        
        prompt = f"Role: {role}\n\nFiles:\n{files_text}\n\nAnalyze:"
        
        # Always allow gradients for embedding computation
        raw_emb = self.llm.embed_observation_with_gradient(
            [prompt], 
            allow_grad=self.training
        )
        
        # Project to RL embedding space
        obs_emb = self.obs_projection(raw_emb)
        return obs_emb
        
    def _select_action(self, role: str, obs_emb: torch.Tensor, hidden: torch.Tensor):
        """Select action and compute all required values"""
        actor = self.actors[role]
        
        # Forward pass through actor
        logits, new_hidden, (latent, mu, sigma) = actor(obs_emb.float(), hidden.float())
        
        # Sample action
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        logprob = dist.log_prob(action)
        
        # Compute value using critic
        value_dict = self.critic(new_hidden)
        value = value_dict[role]
        
        return {
            'action': action.item(),
            'logprob': logprob,
            'value': value,
            'new_hidden': new_hidden,
            'latent': latent,
            'mu': mu,
            'sigma': sigma,
            'logits': logits
        }
        
    def _generate_content(self, role: str, action_idx: int, latent: torch.Tensor) -> str:
        """Generate content using enhanced LLM with latent conditioning"""
        return self.llm.generate_with_latent_conditioning(role, action_idx, latent)
        
    def _run_episode(self) -> Dict[str, float]:
        """Run one complete episode"""
        
        # Reset environment and get batch of problems
        obs_batch = self.env.reset_batch()
        
        episode_reward = 0.0
        episode_length = 0
        final_reward = 0.0
        
        # Reset states and buffers
        device = torch.device(self.cfg.device)
        for role in self.cfg.roles.keys():
            self.hidden_states[role] = torch.zeros(1, self.cfg.model_config.hete_layer_input_dim, 
                                                  device=device, dtype=torch.float32)
            self.buffers[role].reset()
        
        # Store episode data for inference net
        episode_latent_data = []
        
        # Generate solutions for each problem in the batch
        batch_solutions = []
        
        for batch_idx, problem in enumerate(obs_batch):
            # self.console.print(f"[blue]🔧 Processing problem {batch_idx + 1}/{len(obs_batch)}: {problem.get('name', 'Unknown')}[/blue]")
            
            # Generate solution for this problem using multi-agent workflow
            solution = self._generate_solution_for_problem(problem, episode_latent_data)
            batch_solutions.append(solution)
            
            episode_length += 3  # planner, coder, debugger
            self.step += 1
        
        # Submit all solutions to environment at once
        rewards = self.env.step_batch(batch_solutions)
        
        # Calculate episode metrics
        episode_reward = sum(rewards) / len(rewards)
        final_reward = episode_reward
        
        # Distribute rewards to trajectories
        self._distribute_rewards_to_trajectories(rewards)
        
        # Compute GAE for all roles
        self._compute_gae()
        
        return {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "final_reward": final_reward,
            "latent_data": episode_latent_data
        }
    
    def _generate_solution_for_problem(self, problem: Dict[str, Any], episode_latent_data: List) -> str:
        """Generate solution for a single problem using multi-agent workflow - FIXED"""
        
        # Initialize problem workspace
        workspace = {
            "visible_files": {
                "problem.md": f"# {problem.get('name', 'Problem')}\n\n{problem.get('description', 'No description')}"
            }
        }
        
        # Multi-agent workflow: planner -> coder -> debugger
        roles_sequence = ["planner", "coder", "debugger"]
        device = torch.device(self.cfg.device)
        
        for role in roles_sequence:
            # Create role-specific observation
            role_obs = {
                "role": role,
                "visible_files": workspace["visible_files"].copy()
            }
            
            # Process this role
            obs_emb = self._embed_observation(role_obs)
            action_data = self._select_action(role, obs_emb, self.hidden_states[role])
            
            # Update hidden state for this role
            self.hidden_states[role] = action_data['new_hidden']
            
            # Generate content based on role
            content = self._generate_content(role, action_data['action'], action_data['latent'])
            
            # Add generated content to workspace
            if role == "planner":
                filename = "plan.md"
            elif role == "coder":
                filename = "code.py"
            else:  # debugger
                filename = "fixed_code.py"
                
            workspace["visible_files"][filename] = content
            
            # Store trajectory for this role - FIXED: Use detach() to prevent gradient tracking
            traj = Trajectory(
                state=obs_emb.squeeze(0).detach().to(device),
                latent=action_data['latent'].squeeze(0).detach().to(device),
                action=torch.tensor(action_data['action'], device=device),
                logp=action_data['logprob'].detach().to(device),
                reward=torch.tensor(0.0, device=device),
                value=action_data['value'].detach().to(device)
            )
            
            # Store additional data for complete loss computation - FIXED: Use detach()
            traj.mu = action_data['mu'].squeeze(0).detach().to(device)
            traj.sigma = action_data['sigma'].squeeze(0).detach().to(device)
            traj.obs_emb = obs_emb.squeeze(0).detach().to(device)
            
            self.buffers[role].add(traj)
            
            episode_latent_data.append({
                'role': role,
                'mu': action_data['mu'].detach().to(device),
                'sigma': action_data['sigma'].detach().to(device),
                'latent': action_data['latent'].detach().to(device),
                'obs_emb': obs_emb.detach().to(device)
            })
        
        # Return the final code solution
        return workspace["visible_files"]["code.py"]
    
    def _distribute_rewards_to_trajectories(self, rewards: List[float]):
        """Distribute batch rewards to individual trajectories"""
        device = torch.device(self.cfg.device)  # FIXED: Get device
        
        # Each problem generates 3 trajectories (planner, coder, debugger)
        # Distribute each problem's reward to its 3 trajectories
        
        problem_idx = 0
        for role in ["planner", "coder", "debugger"]:
            for traj_idx, traj in enumerate(self.buffers[role].traj):
                current_problem_idx = traj_idx  # Each role has one trajectory per problem
                if current_problem_idx < len(rewards):
                    traj.reward = torch.tensor(rewards[current_problem_idx], device=device)  # FIXED: Specify device
                else:
                    traj.reward = torch.tensor(0.0, device=device)  # FIXED: Specify device
        
    def _compute_gae(self):
        """Compute Generalized Advantage Estimation for all roles"""
        device = torch.device(self.cfg.device)  # FIXED: Get device
        
        for role, buffer in self.buffers.items():
            if len(buffer.traj) == 0:
                continue
                
            rewards = [t.reward.item() for t in buffer.traj]
            values = [t.value.item() for t in buffer.traj]
            
            # GAE computation
            advantages = []
            returns = []
            gae = 0.0
            
            for t in reversed(range(len(rewards))):
                next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
                delta = rewards[t] + self.cfg.gamma * next_value - values[t]
                gae = delta + self.cfg.gamma * self.cfg.lam * gae
                advantages.insert(0, gae)
                returns.insert(0, gae + values[t])
            
            # Normalize advantages
            if len(advantages) > 1:
                mean_adv = sum(advantages) / len(advantages)
                std_adv = (sum((a - mean_adv) ** 2 for a in advantages) / len(advantages)) ** 0.5
                if std_adv > 1e-8:
                    advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]
            
            # Assign to trajectories - FIXED: Ensure device consistency
            for i, traj in enumerate(buffer.traj):
                traj.returns = torch.tensor(returns[i], dtype=torch.float32, device=device)  # FIXED: Specify device
                traj.advs = torch.tensor(advantages[i], dtype=torch.float32, device=device)  # FIXED: Specify device
        
    def _update_networks(self):
        """Update all networks with separate optimizers"""
        total_losses = defaultdict(list)
        
        for epoch in range(self.cfg.n_epochs):
            # 1. Update Actor Networks (PPO loss)
            actor_losses = self._update_actors()
            for k, v in actor_losses.items():
                total_losses[k].extend(v)
            
            # 2. Update Critic Network
            critic_losses = self._update_critic()
            for k, v in critic_losses.items():
                total_losses[k].extend(v)
            
            # 3. Update Latent Networks (Complete SHPPO losses)
            latent_losses = self._update_latent_networks()
            for k, v in latent_losses.items():
                total_losses[k].extend(v)
            
            # 4. Update Inference Network
            inference_losses = self._update_inference_network()
            for k, v in inference_losses.items():
                total_losses[k].extend(v)
            
            # 5. Update LLM (always enabled)
            if self.opt_llm is not None:
                llm_losses = self._update_llm()
                for k, v in llm_losses.items():
                    total_losses[k].extend(v)
                    
        return {k: sum(v) / len(v) if v else 0.0 for k, v in total_losses.items()}
        
    def _update_actors(self) -> Dict[str, List[float]]:
        """Update actor networks with PPO loss - FIXED: Avoid in-place operations"""
        losses = defaultdict(list)
        device = torch.device(self.cfg.device)
        
        for role, buffer in self.buffers.items():
            if len(buffer.traj) == 0:
                continue
                
            actor = self.actors[role]
            optimizer = self.opt_actors[role]
            
            # Process mini-batches
            indices = torch.randperm(len(buffer.traj))
            
            for start in range(0, len(indices), self.cfg.mini_batch_size):
                end = min(start + self.cfg.mini_batch_size, len(indices))
                batch_indices = indices[start:end]
                batch = [buffer.traj[i] for i in batch_indices]
                
                batch_losses = []
                batch_entropies = []
                
                for traj in batch:
                    # FIXED: Create new tensors instead of reusing trajectory tensors
                    # This prevents in-place operations on the same tensor
                    state = traj.state.detach().clone().unsqueeze(0).to(device)
                    action = traj.action.detach().clone().unsqueeze(0).to(device)
                    old_logprob = traj.logp.detach().clone().to(device)
                    advantages = traj.advs.detach().clone().to(device)
                    
                    # Actor forward (no gradient to latent part)
                    with torch.no_grad():
                        mu = traj.mu.detach().clone().unsqueeze(0).to(device)
                        sigma = traj.sigma.detach().clone().unsqueeze(0).to(device)
                        latent_no_grad = mu + sigma * torch.randn_like(sigma)
                    
                    # Only update actor head, not encoder
                    logits = actor.head(state, latent_no_grad)
                    
                    # Policy loss
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    new_logprob = dist.log_prob(action).squeeze()
                    entropy = dist.entropy().squeeze()
                    
                    # PPO clipped loss
                    ratio = torch.exp(new_logprob - old_logprob)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * advantages
                    actor_loss = -torch.min(surr1, surr2)
                    
                    batch_losses.append(actor_loss)
                    batch_entropies.append(entropy)
                
                if batch_losses:
                    # Combine losses
                    total_actor_loss = torch.stack(batch_losses).mean()
                    total_entropy = torch.stack(batch_entropies).mean()
                    final_loss = total_actor_loss - self.cfg.entropy_coeff * total_entropy
                    
                    # Update
                    optimizer.zero_grad()
                    final_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), self.cfg.max_grad_norm)
                    optimizer.step()
                    
                    losses[f'{role}_actor_loss'].append(total_actor_loss.item())
                    losses[f'{role}_entropy'].append(total_entropy.item())
        
        return losses

    def _update_critic(self) -> Dict[str, List[float]]:
        """Update critic network - FIXED: Avoid in-place operations"""
        losses = defaultdict(list)
        device = torch.device(self.cfg.device)
        
        all_states = []
        all_returns = []
        all_roles = []
        
        # Collect all data - FIXED: Use detach().clone() to avoid in-place issues
        for role, buffer in self.buffers.items():
            for traj in buffer.traj:
                all_states.append(traj.obs_emb.detach().clone().to(device))
                all_returns.append(traj.returns.detach().clone().to(device))
                all_roles.append(role)
        
        if not all_states:
            return losses
        
        # Mini-batch updates
        indices = torch.randperm(len(all_states))
        
        for start in range(0, len(indices), self.cfg.mini_batch_size):
            end = min(start + self.cfg.mini_batch_size, len(indices))
            batch_indices = indices[start:end]
            
            batch_states = torch.stack([all_states[i] for i in batch_indices])
            batch_returns = torch.stack([all_returns[i] for i in batch_indices])
            batch_roles = [all_roles[i] for i in batch_indices]
            
            # Forward pass
            value_dict = self.critic(batch_states)
            
            # Compute loss for each role
            total_loss = 0
            for i, role in enumerate(batch_roles):
                predicted_value = value_dict[role][i]
                target_return = batch_returns[i]
                loss = F.mse_loss(predicted_value, target_return)
                total_loss += loss
                losses[f'{role}_critic_loss'].append(loss.item())
            
            # Update
            self.opt_critic.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.critic.parameters()) + list(self.obs_projection.parameters()),
                self.cfg.max_grad_norm
            )
            self.opt_critic.step()
        
        return losses

    def _update_latent_networks(self) -> Dict[str, List[float]]:
        """FIXED: Update latent networks with correct tensor shapes"""
        losses = defaultdict(list)
        device = torch.device(self.cfg.device)
        
        # Collect latent data from all roles
        all_mu = []
        all_sigma = []
        all_latent = []
        all_obs_emb = []
        
        for role, buffer in self.buffers.items():
            for traj in buffer.traj:
                all_mu.append(traj.mu.detach().clone().to(device))
                all_sigma.append(traj.sigma.detach().clone().to(device))
                all_latent.append(traj.latent.detach().clone().to(device))
                all_obs_emb.append(traj.obs_emb.detach().clone().to(device))
        
        if not all_mu:
            return losses
        
        # Stack tensors
        mu_batch = torch.stack(all_mu)        # [batch_size, n_templates, latent_dim]
        sigma_batch = torch.stack(all_sigma)  # [batch_size, n_templates, latent_dim]
        latent_batch = torch.stack(all_latent)
        obs_batch = torch.stack(all_obs_emb)
        
        
        # FIXED: Prepare agent features correctly
        batch_size = mu_batch.size(0)
        n_templates = mu_batch.size(1)
        latent_dim = mu_batch.size(2)
        
        # Combine mu and sigma: [batch_size, n_templates, latent_dim*2]
        agent_features = torch.cat([mu_batch, sigma_batch], dim=-1)
        
        # FIXED: Don't reshape - keep as [batch_size, n_templates, latent_dim*2]
        # The inference network will handle multiple agents (templates) properly
        
        inference_value = self.inference_net(agent_features, obs_batch)
        
        # Compute all latent losses
        latent_losses = self.loss_computer.compute_combined_latent_loss(
            value_estimate=inference_value,
            mu=mu_batch,
            sigma=sigma_batch,
            latent_samples=latent_batch
        )
        
        # Update latent networks
        self.opt_latent.zero_grad()
        latent_losses['combined_latent_loss'].backward()
        
        # Clip gradients
        latent_params = []
        for actor in self.actors.values():
            latent_params.extend(actor.enc.parameters())
        torch.nn.utils.clip_grad_norm_(latent_params, self.cfg.max_grad_norm)
        
        self.opt_latent.step()
        
        # Record losses
        for k, v in latent_losses.items():
            losses[f'latent_{k}'].append(v.item())
        
        return losses

    def _update_inference_network(self) -> Dict[str, List[float]]:
        """FIXED: Update inference network with correct tensor shapes"""
        losses = defaultdict(list)
        device = torch.device(self.cfg.device)
        
        # Collect data
        all_mu = []
        all_sigma = []
        all_obs_emb = []
        all_returns = []
        
        for role, buffer in self.buffers.items():
            for traj in buffer.traj:
                all_mu.append(traj.mu.detach().clone().to(device))
                all_sigma.append(traj.sigma.detach().clone().to(device))
                all_obs_emb.append(traj.obs_emb.detach().clone().to(device))
                all_returns.append(traj.returns.detach().clone().to(device))
        
        if not all_mu:
            return losses
        
        # Stack and prepare data
        mu_batch = torch.stack(all_mu)
        sigma_batch = torch.stack(all_sigma)
        obs_batch = torch.stack(all_obs_emb)
        returns_batch = torch.stack(all_returns)
        
        # FIXED: Prepare for inference net with correct dimensions
        agent_features = torch.cat([mu_batch, sigma_batch], dim=-1)
        # Keep as [batch_size, n_templates, latent_dim*2] - don't reshape
        
        # Forward pass
        predicted_values = self.inference_net(agent_features, obs_batch)
        
        # MSE loss
        inference_loss = F.mse_loss(predicted_values, returns_batch)
        
        # Update
        self.opt_inference.zero_grad()
        inference_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.inference_net.parameters(), self.cfg.max_grad_norm)
        self.opt_inference.step()
        
        losses['inference_mse_loss'].append(inference_loss.item())
        
        return losses

    def _update_llm(self) -> Dict[str, List[float]]:
        """BEST FIX: Update LLM parameters with proper gradient tracking"""
        losses = defaultdict(list)
        device = torch.device(self.cfg.device)
        
        # Skip LLM update if no trajectories
        if not any(len(buffer.traj) > 0 for buffer in self.buffers.values()):
            return losses
        
        # Generate new embeddings with gradients for LLM training
        sample_prompts = []
        for role in ["planner", "coder", "debugger"]:
            sample_prompts.append(f"Role: {role}\n\nAnalyze and generate code solution")
        
        # Create fresh embeddings with gradient tracking
        raw_embeddings = self.llm.embed_observation_with_gradient(
            sample_prompts,
            allow_grad=True  # Essential for LLM parameter updates
        )
        
        # Project to RL embedding space
        projected_embeddings = self.obs_projection(raw_embeddings)
        
        # Compute diversity loss to encourage varied representations
        if projected_embeddings.size(0) > 1:
            # Use standard deviation as diversity measure
            std_dev = torch.std(projected_embeddings, dim=0).mean()
            diversity_loss = -std_dev  # Maximize diversity
        else:
            # Single sample: use L2 regularization
            diversity_loss = torch.mean(projected_embeddings ** 2) * 0.01
        
        # Update LLM LoRA parameters
        self.opt_llm.zero_grad()
        diversity_loss.backward()
        
        # Clip gradients
        llm_params = self.llm.trainable_parameters()
        if llm_params:
            torch.nn.utils.clip_grad_norm_(llm_params, self.cfg.max_grad_norm)
        
        self.opt_llm.step()
        
        losses['llm_diversity_loss'].append(diversity_loss.item())
        return losses
        
    def train(self):
        """Main training loop with Rich interface"""
        self.console.print(Panel(
            f"[bold blue]🚀 Training Complete SHPPO for {self.cfg.n_episodes} episodes[/bold blue]",
            title="Training Started"
        ))
        
        # Rich progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            train_task = progress.add_task(
                "[green]Training Episodes...", 
                total=self.cfg.n_episodes
            )
            
            for episode in range(self.cfg.n_episodes):
                self.episode = episode
                
                # Reset buffers
                for buffer in self.buffers.values():
                    buffer.reset()
                
                # Run episode
                episode_metrics = self._run_episode()
                
                # Update all networks with separate optimizers
                loss_metrics = self._update_networks()
                
                # Calculate metrics
                episode_reward = episode_metrics.get('episode_reward', 0.0)
                pass_rate = 1.0 if episode_reward >= 0.99 else 0.0
                
                # Update best reward
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                
                # Logging
                log_data = {
                    "train/episode": episode,
                    "train/episode_reward": episode_reward,
                    "train/pass_rate": pass_rate,
                    "train/episode_length": episode_metrics.get('episode_length', 0),
                    "train/best_reward": self.best_reward,
                }
                
                # Add all loss metrics
                for k, v in loss_metrics.items():
                    log_data[f"train/{k}"] = v
                
                wandb.log(log_data, step=episode)
                
                # Update progress bar
                progress.update(
                    train_task, 
                    advance=1,
                    description=f"[green]Episode {episode}: Reward={episode_reward:.3f}, Pass={pass_rate:.3f}"
                )
                
                # Detailed logging with Rich
                if episode % self.cfg.log_interval == 0:
                    self._log_episode_details(episode, episode_reward, pass_rate, loss_metrics)
                
                # Save checkpoint
                if episode > 0 and episode % self.cfg.save_interval == 0:
                    self._save_checkpoint(episode)
                    
        self.console.print(Panel(
            "[bold green]🎉 Training completed successfully![/bold green]\n"
            f"• Total episodes: {self.cfg.n_episodes}\n"
            f"• Best reward: {self.best_reward:.4f}",
            title="Training Complete"
        ))
        wandb.finish()
    
    def _log_episode_details(self, episode: int, reward: float, pass_rate: float, losses: Dict[str, float]):
        """Log detailed episode information with Rich tables"""
        
        # Create metrics table
        table = Table(title=f"Episode {episode} Metrics")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        table.add_column("Type", style="green")
        
        # Add main metrics
        table.add_row("Episode Reward", f"{reward:.4f}", "Performance")
        table.add_row("Pass Rate", f"{pass_rate:.3f}", "Performance")
        table.add_row("Best Reward", f"{self.best_reward:.4f}", "Performance")
        
        # Add loss metrics
        actor_losses = {k: v for k, v in losses.items() if 'actor_loss' in k}
        latent_losses = {k: v for k, v in losses.items() if 'latent_' in k}
        other_losses = {k: v for k, v in losses.items() if k not in actor_losses and k not in latent_losses}
        
        for k, v in actor_losses.items():
            table.add_row(k.replace('_', ' ').title(), f"{v:.6f}", "Actor Loss")
        
        for k, v in latent_losses.items():
            table.add_row(k.replace('_', ' ').title(), f"{v:.6f}", "Latent Loss")
            
        for k, v in other_losses.items():
            table.add_row(k.replace('_', ' ').title(), f"{v:.6f}", "Other Loss")
        
        self.console.print(table)
        
        # Performance indicator
        if pass_rate >= 0.8:
            status = "[bold green]🎯 Excellent Performance![/bold green]"
        elif pass_rate >= 0.5:
            status = "[yellow]⚡ Good Progress[/yellow]"
        elif pass_rate >= 0.1:
            status = "[orange3]🔄 Learning...[/orange3]"
        else:
            status = "[red]💪 Keep Training[/red]"
        
        self.console.print(Panel(status, title="Status"))
        
    def _save_checkpoint(self, episode):
        """Save checkpoint with Rich logging"""
        checkpoint = {
            'episode': episode,
            'actors': {role: actor.state_dict() for role, actor in self.actors.items()},
            'critic': self.critic.state_dict(),
            'inference_net': self.inference_net.state_dict(),
            'obs_projection': self.obs_projection.state_dict(),
            'llm': self.llm.model.state_dict(),  # Save LLM state too
            'config': self.cfg,
            'metrics': dict(self.metrics),
            'best_reward': self.best_reward
        }
        
        path = f"{self.cfg.checkpoint_dir}/checkpoint_{episode}.pt"
        torch.save(checkpoint, path)
        
        self.console.print(f"[blue]💾 Checkpoint saved: {path}[/blue]")

def main():
    """Main training function with Rich interface"""
    import argparse
    
    console = Console()
    
    parser = argparse.ArgumentParser(description="Complete SHPPO Training")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes")
    parser.add_argument("--device", type=str, default="auto", help="Training device")
    parser.add_argument("--wandb_project", type=str, default="shppo-complete", help="WandB project name")
    parser.add_argument("--batch_size", type=int, default=4, help="Environment batch size")
    args = parser.parse_args()
    
    # Create config
    config = SHPPOConfig()
    config.n_episodes = args.episodes
    config.wandb_project = args.wandb_project
    config.env_config.batch_size = args.batch_size
    
    if args.device != "auto":
        config.device = args.device
    
    # Display startup info
    console.print(Panel(
        f"[bold cyan]Complete SHPPO Training[/bold cyan]\n"
        f"• Episodes: {config.n_episodes}\n"
        f"• Device: {config.device}\n"
        f"• Batch Size: {config.env_config.batch_size}\n"
        f"• LLM Training: Always Enabled 🔥\n"
        f"• WandB Project: {config.wandb_project}",
        title="🧠 Training Configuration"
    ))
    
    # Create trainer
    trainer = CompleteSHPPOTrainer(config)
    trainer.train()
    console.print("[bold green]✅ Training completed successfully![/bold green]")

if __name__ == "__main__":
    main()